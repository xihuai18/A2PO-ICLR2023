import time
from functools import reduce
from pprint import pprint

import numpy as np
import torch

from onpolicy.runner.base_runner import Runner
from onpolicy.utils.util import _t2n


class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = (int(self.num_env_steps) // self.episode_length //
                    self.n_rollout_threads)

        episode_lens = []
        one_episode_len = np.zeros(self.n_rollout_threads, dtype=np.int)

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

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
                    available_actions,
                ) = self.envs.step(actions)

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

                one_episode_len += 1

                done_env = np.all(dones, axis=1)

                for i in range(self.n_rollout_threads):
                    if done_env[i]:
                        episode_lens.append(one_episode_len[i].copy())
                        one_episode_len[i] = 0

            # NOTE compute return and update network
            # if using trace techniques, the advantage is computed when updating the policy (except the first ppo epoch)
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
                    "\n Map {} Algo {} Exp {} Seed {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(
                        self.all_args.map_name,
                        self.algorithm_name,
                        self.experiment_name,
                        self.all_args.seed,
                        episode * self.n_rollout_threads,
                        episodes * self.n_rollout_threads,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    ))

                if self.env_name == "StarCraft2":
                    battles_won = []
                    battles_game = []
                    train_battles_won = []
                    train_battles_game = []

                    for i, info in enumerate(infos):
                        if "battles_won" in info[0].keys():
                            battles_won.append(info[0]["battles_won"])
                            train_battles_won.append(info[0]["battles_won"] -
                                                     last_battles_won[i])
                        if "battles_game" in info[0].keys():
                            battles_game.append(info[0]["battles_game"])
                            train_battles_game.append(info[0]["battles_game"] -
                                                      last_battles_game[i])

                    train_win_rate = (np.sum(train_battles_won) /
                                      np.sum(train_battles_game) if
                                      np.sum(train_battles_game) > 0 else 0.0)
                    self.logger.log_stat("train_win_rate", train_win_rate,
                                         total_num_steps)

                    average_episode_len = np.mean(
                        episode_lens) if len(episode_lens) > 0 else 0.0
                    episode_lens = []

                    self.logger.log_stat("average_episode_length",
                                         average_episode_len, total_num_steps)

                    print(
                        "train games {:.4f} train win rate is {:.4f} train average step reward is {:.4f} average episode length is {:.4f}."
                        .format(np.sum(train_battles_game), train_win_rate,
                                np.mean(self.buffer.rewards),
                                average_episode_len))

                    last_battles_game = battles_game
                    last_battles_won = battles_won

                train_infos["dead_ratio"] = 1 - self.buffer.active_masks.sum(
                ) / reduce(lambda x, y: x * y,
                           list(self.buffer.active_masks.shape))

                self.log_train(train_infos, total_num_steps)

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
            rnn_state,
            rnn_state_critic,
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
            available_actions,
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
            available_actions,
        )

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            self.logger.log_stat(k, v, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = np.zeros(self.n_eval_rollout_threads)

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset(
        )

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
                np.concatenate(eval_available_actions),
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
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            one_episode_rewards += eval_rewards.mean(1).reshape(-1)

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
                    eval_episode_rewards.append(
                        one_episode_rewards[eval_i].copy())
                    one_episode_rewards[eval_i] = 0.0
                    if eval_infos[eval_i][0]["won"]:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {
                    "eval_average_episode_return": eval_episode_rewards
                }
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won / eval_episode
                print(
                    "eval win rate is {:.4f} average episode return is {:.4f}."
                    .format(eval_win_rate, np.mean(eval_episode_rewards)))
                self.logger.log_stat("eval_win_rate",
                                     eval_win_rate,
                                     total_num_steps,
                                     eval_stat=True)
                break
