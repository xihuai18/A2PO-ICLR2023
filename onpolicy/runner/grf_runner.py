import time

import numpy as np
import torch
import tqdm
from onpolicy.runner.base_runner import Runner
from prettytable import PrettyTable


def _t2n(x):
    return x.detach().cpu().numpy()


class GRFRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(GRFRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        cum_episode_return = np.zeros(self.n_rollout_threads, dtype=np.float32)
        cum_episode_len = np.zeros(self.n_rollout_threads, dtype=np.int32)

        train_episode_return = []
        train_episode_len = []
        train_episode_goal = []
        train_episode_goal_diff = []
        train_episode_win = []
        train_episode_yellow = []
        train_episode_active = []

        train_episode_stats_info = {}

        # train_episode_shot = []
        # train_episode_pass = []

        if self.all_args.action_set == "v2":
            cum_episode_builtin_ai_count = np.zeros(
                self.n_rollout_threads, dtype=np.float
            )
            train_builtin_ai_count = []

        tqdm_bar = tqdm.tqdm(
            range(episodes),
            position=0,
            desc="ep",
            leave=False,
            bar_format="{l_bar}{r_bar}",
        )
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # for step in tqdm.tqdm(range(self.episode_length), position=1, leave=False, desc="step"):
            # for step in tqdm.tqdm(range(self.episode_length),
            #                       position=1,
            #                       desc="step"):
            for step in range(self.episode_length):
                tqdm_bar.set_description(f"ep {episode} step {step}")
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)
                if self.all_args.action_set == "v2":
                    cum_episode_builtin_ai_count += np.mean(
                        (np.vstack(actions_env) == 19)
                        .reshape(self.all_args.n_rollout_threads, self.num_agents)
                        .astype(np.float),
                        axis=-1,
                    )

                # Obser reward and next obs
                (obs, share_obs, rewards, dones, infos, avail) = self.envs.step(
                    actions_env
                )

                cum_episode_return += np.mean(rewards, axis=1).reshape(-1)
                cum_episode_len += 1

                _done = np.all(dones, axis=1)
                # print(step, _done[0], infos[0][0]["steps_left"])

                for t in range(self.all_args.n_rollout_threads):
                    if _done[t]:
                        # print("done step", step)
                        train_episode_return.append(cum_episode_return[t].copy())
                        train_episode_len.append(cum_episode_len[t].copy())

                        my_score, opp_score = infos[t][0]["score"]
                        train_episode_goal.append(my_score)
                        train_episode_goal_diff.append(my_score - opp_score)
                        train_episode_win.append(1 if my_score > opp_score else 0)
                        train_episode_yellow.append(
                            sum(infos[t][0]["left_team_yellow_card"])
                        )
                        train_episode_active.append(
                            sum(infos[t][0]["left_team_active"])
                        )

                        if self.all_args.action_set == "v2":
                            train_builtin_ai_count.append(
                                cum_episode_builtin_ai_count[t] / cum_episode_len[t]
                            )
                            cum_episode_builtin_ai_count[t] = 0

                        # print(infos[t][0])
                        total_num_steps = (
                            (episode + 1) * self.episode_length * self.n_rollout_threads
                        )
                        if len(train_episode_stats_info) == 0:
                            for k, v in infos[t][0]["stats"].items():
                                if "cood" in k:
                                    self.logger.get_sacred_run().log_scalar("cood pair", v, total_num_steps)
                                else:
                                    train_episode_stats_info[k] = [v]
                        else:
                            for k, v in infos[t][0]["stats"].items():
                                if "cood" in k:
                                    self.logger.get_sacred_run().log_scalar("cood pair", v, total_num_steps)
                                else:
                                    train_episode_stats_info[k].append(v)

                        cum_episode_return[t] = 0.0
                        cum_episode_len[t] = 0

                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    avail,
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

            train_infos["average_episode_return"] = (
                np.mean(train_episode_return) if len(train_episode_return) > 0 else 0.0
            )
            train_infos["average_episode_len"] = (
                np.mean(train_episode_len) if len(train_episode_len) > 0 else 0.0
            )
            train_infos["average_episode_goal"] = (
                np.mean(train_episode_goal) if len(train_episode_goal) > 0 else 0.0
            )
            train_infos["average_episode_goal_diff"] = (
                np.mean(train_episode_goal_diff)
                if len(train_episode_goal_diff) > 0
                else 0.0
            )
            train_infos["average_episode_win_rate"] = (
                np.mean(train_episode_win) if len(train_episode_win) > 0 else 0.0
            )
            train_infos["average_episode_yellow_card"] = (
                np.mean(train_episode_yellow) if len(train_episode_yellow) > 0 else 0.0
            )
            train_infos["average_episode_active"] = (
                np.mean(train_episode_active) if len(train_episode_active) > 0 else 0.0
            )
            for k, v in train_episode_stats_info.items():
                train_infos[k.replace(" ", "_")] = np.mean(v) if len(v) > 0 else 0.0
            if self.all_args.action_set == "v2":
                train_infos["average_builtin_ai_count"] = (
                    np.mean(train_builtin_ai_count)
                    if len(train_builtin_ai_count) > 0
                    else 0.0
                )

            # post process
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )
            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0 or episode == episodes - 1:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} Seed {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        self.all_args.seed,
                        episode * self.n_rollout_threads,
                        episodes * self.n_rollout_threads,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                if self.env_name == "grf":
                    table = PrettyTable(["Stat", "Value"])
                    table.align = "l"
                    for field in table.field_names:
                        table.float_format[field] = ".4f"
                    stats = [
                        "Num of Games",
                        "Average Reward",
                        "Average Episode Return",
                        "Average Episode Length",
                        "Average Episode Goal",
                        "Average Episode Goal Diff",
                        "Average Episode Win Rate",
                        "Average Episode Yellow Card",
                        "Average Episode Active",
                    ]
                    values = [
                        len(train_episode_len),
                        np.mean(self.buffer.rewards),
                        train_infos["average_episode_return"],
                        train_infos["average_episode_len"],
                        train_infos["average_episode_goal"],
                        train_infos["average_episode_goal_diff"],
                        train_infos["average_episode_win_rate"],
                        train_infos["average_episode_yellow_card"],
                        train_infos["average_episode_active"],
                        train_infos["average_episode_active"],
                    ]

                    rows = list(zip(stats, values))
                    table.add_rows(rows)
                    if self.all_args.action_set == "v2":
                        table.add_row(
                            [
                                "Average Builtin AI Count",
                                train_infos["average_builtin_ai_count"],
                            ]
                        )
                    for k in sorted(train_episode_stats_info.keys()):
                        table.add_row([k, train_infos[k.replace(" ", "_")]])
                    print(table)
                    print()
                    table.clear_rows()

                self.log_train(train_infos, total_num_steps)
                train_episode_return = []
                train_episode_len = []
                train_episode_goal = []
                train_episode_goal_diff = []
                train_episode_win = []
                train_episode_yellow = []
                train_episode_active = []
                train_episode_stats_info = {}
                if self.all_args.action_set == "v2":
                    train_builtin_ai_count = []
            # eval
            if (
                episode % self.eval_interval == 0 or episode == episodes - 1
            ) and self.use_eval:
                self.eval(total_num_steps)

            tqdm_bar.update()
        tqdm_bar.close()

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        # print(obs)
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
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_state_critic), self.n_rollout_threads)
        )

        # rearrange action
        actions_env = [actions[idx, :, 0] for idx in range(self.n_rollout_threads)]

        # print("actions_env shape", [a_env.shape for a_env in actions_env])

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

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        if not self.use_centralized_V:
            share_obs = obs

        bad_masks = np.array(
            [
                [
                    [0.0] if info[agent_id]["bad_transition"] else [1.0]
                    for agent_id in range(self.num_agents)
                ]
                for info in infos
            ]
        )

        # for info in infos:
        #     if info[0]["bad_transition"]:
        #         print(info)

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
            available_actions=available_actions,
        )

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            self.logger.log_stat(k, v, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0

        eval_episode_return = []
        eval_episode_goal_diffs = []
        eval_episode_goals = []
        eval_episode_win = []
        eval_episode_stat_info = {}

        one_episode_return = np.zeros((self.n_eval_rollout_threads))

        eval_obs, _, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        tqdm_bar = tqdm.trange(
            self.all_args.eval_episodes,
            desc="eval ep",
            position=1,
            leave=False,
            bar_format="{l_bar}{r_bar}",
        )
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
                np.split(_t2n(eval_actions), self.n_eval_rollout_threads)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
            )

            eval_actions_env = [
                eval_actions[idx, :, 0] for idx in range(self.n_eval_rollout_threads)
            ]

            # Obser reward and next obs
            (
                eval_obs,
                _,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions_env)
            # print(eval_rewards)
            one_episode_return += np.mean(eval_rewards, axis=1).reshape(-1)
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
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    tqdm_bar.update()

                    eval_episode_return.append(one_episode_return[eval_i].copy())
                    one_episode_return[eval_i] = 0

                    my_score, opp_score = eval_infos[eval_i][0]["score"]
                    eval_episode_goals.append(my_score)
                    eval_episode_goal_diffs.append(my_score - opp_score)

                    if len(eval_episode_stat_info) == 0:
                        for k, v in eval_infos[eval_i][0]["stats"].items():
                            if "cood" not in k:
                                eval_episode_stat_info[k] = [v]
                    else:
                        for k, v in eval_infos[eval_i][0]["stats"].items():
                            if "cood" not in k:
                                eval_episode_stat_info[k].append(v)

                    eval_episode_win.append(1 if my_score > opp_score else 0)

            if eval_episode >= self.all_args.eval_episodes:
                tqdm_bar.close()
                # print(eval_episode_return)
                eval_episode_return = np.array(eval_episode_return)
                eval_env_infos = {
                    "eval_average_episode_return": np.mean(eval_episode_return),
                    "eval_average_episode_goals": np.mean(eval_episode_goals),
                    "eval_average_episode_goal_diffs": np.mean(eval_episode_goal_diffs),
                    "eval_average_episode_win_rate": np.mean(eval_episode_win),
                }

                for k, v in eval_episode_stat_info.items():
                    eval_env_infos[k.replace(" ", "_")] = np.mean(v)
                table = PrettyTable(["Eval Stat", "Value"])
                table.align = "l"
                for field in table.field_names:
                    table.float_format[field] = ".4f"
                rows = [(k, eval_env_infos[k]) for k in sorted(eval_env_infos.keys())]
                table.add_rows(rows)
                # print(
                #     f'evaluation {eval_episode} episodes:\n\taverage episode return {eval_env_infos["eval_average_episode_return"]:.4f}\n\taverage goals {eval_env_infos["eval_average_episode_goals"]:.4f}\n\taverage goal diffs {eval_env_infos["eval_average_episode_goal_diffs"]:.4f}\n\taverage win rate {eval_env_infos["eval_average_episode_win_rate"]:.4f}'
                # )
                print(table)
                table.clear_rows()
                self.log_env(eval_env_infos, total_num_steps)
                break
