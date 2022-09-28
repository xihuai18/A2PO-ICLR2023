import time

import numpy as np
import torch

from onpolicy.runner.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class PettingZooRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(PettingZooRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = (int(self.num_env_steps) // self.episode_length //
                    self.n_rollout_threads)
        episode_return = []
        one_episode_return = np.zeros(
            (self.all_args.n_rollout_threads, self.num_agents))

        episode_lens = []
        one_episode_len = np.zeros(self.all_args.n_rollout_threads)

        for episode in range(episodes):

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(
                    actions_env)

                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

                one_episode_return += rewards.reshape(-1, self.num_agents)
                one_episode_len += 1

                thread_done = np.all(dones, axis=1)
                for t_i in range(self.n_rollout_threads):
                    if thread_done[t_i]:
                        episode_return.append(one_episode_return[t_i].copy())
                        one_episode_return[t_i] = np.zeros(self.num_agents)
                        episode_lens.append(one_episode_len[t_i].copy())
                        one_episode_len[t_i] = 0

            # compute return and update network
            self.compute()
            train_infos = self.train(episode)

            # post process
            total_num_steps = ((episode + 1) * self.episode_length *
                               self.n_rollout_threads)

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0 or episode == episodes - 1:
                end = time.time()
                print(
                    "\n Scenario {} Num {} Algo {} Exp {} Seed {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(
                        self.all_args.scenario_name,
                        self.all_args.num_agents,
                        self.algorithm_name,
                        self.experiment_name,
                        self.all_args.seed,
                        episode * self.n_rollout_threads,
                        episodes * self.n_rollout_threads,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    ))

                # print(_episode_return.mean(axis=1))
                _episode_return = np.array(episode_return)

                if self.env_name == "PettingZoo":
                    env_infos = {}
                    if len(episode_return) > 0:
                        for agent_id in range(self.num_agents):
                            agent_k = "agent%i/individual_rewards" % agent_id
                            env_infos[agent_k] = np.mean(
                                _episode_return[:, agent_id]
                            ) if len(_episode_return) > 0 else 0.0

                average_len = np.mean(
                    episode_lens) if len(episode_lens) > 0 else 0.0
                train_infos["average_episode_rewards"] = np.mean(
                    _episode_return) if _episode_return.shape[0] > 0 else 0.0
                train_infos["average_episode_lens"] = average_len
                print("average episode rewards is {} length is {}".format(
                    train_infos["average_episode_rewards"],
                    train_infos["average_episode_lens"]))
                self.log_train(train_infos, total_num_steps)
                self.log_train(env_infos, total_num_steps)
                episode_return = []
                episode_lens = []

            # eval
            if (episode % self.eval_interval == 0
                    or episode == episodes - 1) and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            np.concatenate(self.buffer.available_actions[step]),
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(
            np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action

        if self.envs.action_space[0].__class__.__name__ == "Box":
            actions_env = actions
        elif self.envs.action_space[0].__class__.__name__ == "Discrete":
            actions_env = actions.squeeze(-1)
        else:
            raise NotImplementedError
        # print(actions_env)

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
                        dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1),
                                        dtype=np.float32)

        bad_masks = np.array(
            [[[0.0] if info[agent_id]["bad_transition"] else [1.0]
              for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs,
                           obs,
                           rnn_states,
                           rnn_states_critic,
                           actions,
                           action_log_probs,
                           values,
                           rewards,
                           masks,
                           bad_masks,
                           available_actions=available_actions)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_returns = []
        one_episode_rewards = np.zeros((self.n_eval_rollout_threads))
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset(
        )

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1),
                             dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                np.concatenate(eval_available_actions),
                deterministic=True,
            )
            eval_actions = np.array(
                np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.envs.action_space[0].__class__.__name__ == "Box":
                eval_actions_env = eval_actions
            elif self.envs.action_space[0].__class__.__name__ == "Discrete":
                eval_actions_env = eval_actions.squeeze(-1)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions_env)

            one_episode_rewards += np.mean(eval_rewards, axis=1).reshape(-1)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones
                  == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1),
                dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32)

            eval_done_env = np.all(eval_dones, axis=1)
            for t_i in range(self.n_eval_rollout_threads):
                if eval_done_env[t_i]:
                    eval_episode_returns.append(one_episode_rewards[t_i])
                    one_episode_rewards[t_i] = 0.0
            if len(eval_episode_returns) >= self.all_args.eval_episodes:
                break

        eval_episode_returns = np.array(eval_episode_returns)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = eval_episode_returns
        eval_average_episode_return = np.mean(
            eval_env_infos["eval_average_episode_rewards"])
        print("eval average episode rewards of agent: " +
              str(eval_average_episode_return))
        self.log_env(eval_env_infos, total_num_steps)
