import time
import numpy as np
import torch
from onpolicy.runner.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class MujocoRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(MujocoRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = (int(self.num_env_steps) // self.episode_length //
                    self.n_rollout_threads)

        cum_episode_return = np.zeros(self.n_rollout_threads, dtype=np.float32)
        cum_episode_len = np.zeros(self.n_rollout_threads, dtype=np.int32)

        train_episode_return = []
        train_episode_len = []
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
                ) = self.collect(step)

                # Obser reward and next obs
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    _,
                ) = self.envs.step(actions)

                cum_episode_return += np.mean(rewards, axis=1).reshape(-1)
                cum_episode_len += 1

                _done = np.all(dones, axis=1)

                for t in range(self.all_args.n_rollout_threads):
                    if _done[t]:
                        train_episode_return.append(cum_episode_return[t].copy())
                        train_episode_len.append(cum_episode_len[t].copy())
                        cum_episode_return[t] = 0.0
                        cum_episode_len[t] = 0

                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            # NOTE compute return and update network
            # if using trace techniques, the advantage is computed when updating the policy (except the first ppo epoch)
            self.compute()
            train_infos = self.train(episode)

            train_infos["average_episode_return"] = np.mean(
                train_episode_return) if len(train_episode_return) > 0 else 0.0
            train_infos["average_episode_len"] = np.mean(train_episode_len) if len(train_episode_len) > 0 else 0.0

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
                    "\n Scenario {} Conf {} Algo {} Exp {} Seed {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(
                        self.all_args.scenario_name,
                        self.all_args.agent_conf,
                        self.algorithm_name,
                        self.experiment_name,
                        self.all_args.seed,
                        episode * self.n_rollout_threads,
                        episodes * self.n_rollout_threads,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    ))

                if self.env_name == "mujoco":
                    print(
                        "train games {:.4f} train episode return is {:.4f} train average step reward is {:.4f} average episode length is {:.4f}."
                        .format(len(train_episode_len),
                                train_infos["average_episode_return"],
                                np.mean(self.buffer.rewards),
                                train_infos["average_episode_len"]))

                self.log_train(train_infos, total_num_steps)
                train_episode_return = []
                train_episode_len = []
            # eval
            if (episode % self.eval_interval == 0 or episode == episodes -1) and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        # print(obs)
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_state,
            rnn_state_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state),
                                       self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        rnn_states_critic[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                *self.buffer.rnn_states_critic.shape[3:],
            ),
            dtype=np.float32,
        )

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
                        dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
                               dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1),
                                               dtype=np.float32)
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array(
            [[[0.0] if info[agent_id]["bad_transition"] else [1.0]
              for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            bad_masks,
            active_masks,
        )

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            self.logger.log_stat(k, v, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0

        eval_episode_lens = []
        one_episode_lens = np.zeros((self.n_eval_rollout_threads))
        eval_episode_rewards = []
        one_episode_rewards = np.zeros((self.n_eval_rollout_threads))

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1),
                             dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(
                np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                _,
            ) = self.eval_envs.step(eval_actions)
            # print(eval_rewards)
            one_episode_rewards += np.mean(eval_rewards, axis=1).reshape(-1)
            one_episode_lens += 1
            # print("reward", eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (self.all_args.n_eval_rollout_threads, self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1),
                dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(one_episode_rewards[eval_i].copy())
                    one_episode_rewards[eval_i] = 0

                    eval_episode_lens.append(one_episode_lens[eval_i].copy())
                    one_episode_lens[eval_i] = 0

            if eval_episode >= self.all_args.eval_episodes:
                # print(eval_episode_rewards)
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {
                    "eval_average_episode_return":
                    eval_episode_rewards,
                    "eval_average_episode_len":
                    eval_episode_lens,
                    "eval_average_episode_reward":
                    np.mean(eval_episode_rewards) / np.mean(eval_episode_lens)
                }
                print(
                    f"eval average episode return {np.mean(eval_episode_rewards):.4f} average length {np.mean(eval_episode_lens):.4f} average reward {np.mean(eval_episode_rewards) / np.mean(eval_episode_lens):.4f}"
                )
                self.log_env(eval_env_infos, total_num_steps)
                break
            # print("eval_episode:", eval_episode)
