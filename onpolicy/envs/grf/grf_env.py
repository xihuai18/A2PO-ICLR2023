from os import stat
import random
from pathlib import Path
from typing import Dict, List

import gfootball.env as football_env
import numpy as np
from gym import spaces

from .multiagentenv import MultiAgentEnv
from .raw_feature_process import FeatureEncoder
from .reward_process import Rewarder
from .stats_process import StatsObserver


class FootballEnv(MultiAgentEnv):
    """Wrapper to make Google Research Football environment compatible"""

    def __init__(self, args, evaluation: bool = False, seed: int = None):
        self.num_agents = args.num_agents
        self.scenario_name = args.scenario_name
        self.representation = args.representation
        self.share_reward = args.share_reward
        self.reward_config = args.reward_config
        self.reward_shaping = args.reward_shaping
        self.obs_last_action = args.obs_last_action

        # make env
        if evaluation:
            self.env = football_env.create_environment(
                env_name=args.scenario_name,
                stacked=args.use_stacked_frames,
                representation=args.representation,
                rewards=args.rewards,
                number_of_left_players_agent_controls=args.num_agents,
                number_of_right_players_agent_controls=0,
                channel_dimensions=(args.smm_width, args.smm_height),
                other_config_options={
                    "action_set": args.action_set,
                    "game_engine_random_seed": seed,
                },
                # render=True,
                # write_video=True,
                # write_full_episode_dumps=True,
                # dump_frequency=1,
                # logdir=Path(args.exp_dir) / "replays",
            )
            print("Evaluation mode GRF")
        else:
            self.env = football_env.create_environment(
                env_name=args.scenario_name,
                stacked=args.use_stacked_frames,
                representation=args.representation,
                rewards=args.rewards,
                number_of_left_players_agent_controls=args.num_agents,
                number_of_right_players_agent_controls=0,
                channel_dimensions=(args.smm_width, args.smm_height),
                other_config_options={
                    "action_set": args.action_set,
                    "game_engine_random_seed": seed,
                },
            )

        self.action_n = self.env.action_space[0].n
        self.num_left_agents = len(list(self.env.unwrapped._env._env.config.left_team))
        self.num_right_agents = len(
            list(self.env.unwrapped._env._env.config.right_team)
        )
        assert self.num_agents <= self.num_left_agents

        # assert (self.env.unwrapped._env._env.config.game_duration + 1) == args.episode_length, f"{self.env.unwrapped._env._env.config.game_duration} != {args.episode_length}"

        self.max_steps = self.env.unwrapped.observation()[0]["steps_left"]

        self.feature_encoder = FeatureEncoder(
            self.num_left_agents,
            self.num_right_agents,
            self.action_n,
            args.avail_in_feature,
            args.obs_agent_id,
            args.obs_last_action,
            args.obs_match,
            game_length=args.episode_length,
        )
        self.reward_encoder = Rewarder(
            args.reward_config if args.reward_shaping else None
        )
        player_ids = [obs["active"] for obs in self.env.unwrapped.observation()]
        self.statsobserver = StatsObserver(self.num_agents, player_ids)

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        # print(f"{self.num_left_agents} vs {self.num_right_agents}")
        if self.num_agents == 1:
            if "simple" in self.representation:
                obs_space = self.env.observation_space
            else:
                obs_space = self.feature_encoder.observation_space
            self.action_space.append(self.env.action_space)
            self.observation_space.append(obs_space)
            self.share_observation_space.append(obs_space)
        else:
            for idx in range(self.num_agents):
                self.action_space.append(
                    spaces.Discrete(n=self.env.action_space[idx].n)
                )
                if "simple" in self.representation:
                    obs_space = spaces.Box(
                        low=self.env.observation_space.low[idx],
                        high=self.env.observation_space.high[idx],
                        shape=self.env.observation_space.shape[1:],
                        dtype=self.env.observation_space.dtype,
                    )
                else:
                    obs_space = self.feature_encoder.observation_space
                # print(obs_space.shape)
                self.observation_space.append(obs_space)
                self.share_observation_space.append(obs_space)
        self.prev_obs = None
        if self.obs_last_action:
            self.last_action = np.zeros((self.num_agents, self.action_n))

    def encode_obs_agent(self, obs: Dict) -> "obs, avail":
        if "raw" in self.representation:
            return self.feature_encoder.encode_each(obs)
        else:
            return obs, np.ones(self.action_n)

    def encode_obs(self, obs: List[Dict]):
        obs_list = []
        avail_list = []
        for _i, _obs in enumerate(obs):
            _obs, _avail = self.encode_obs_agent(_obs)
            if self.obs_last_action:
                _obs = np.concatenate((_obs, self.last_action[_i]), axis=-1)
            obs_list.append(_obs)
            avail_list.append(_avail)
        return obs_list, avail_list

    def reset(self):
        obs = self.env.reset()
        # print("reset", obs[0]["steps_left"])
        self.prev_obs = obs.copy()
        obs, avail = self.encode_obs(obs)
        self.statsobserver.reset()

        return obs, obs.copy(), avail

    def step(self, action):
        if self.obs_last_action:
            self.last_action = np.eye(self.action_n)[np.array(action)]
        obs, reward, done, info = self.env.step(action)
        reward = reward.reshape(self.num_agents, 1)
        for idx in range(self.num_agents):
            reward[idx] = self.reward_encoder.calc_reward(
                reward[idx], self.prev_obs[idx], obs[idx]
            )
        self.prev_obs = obs.copy()
        self.statsobserver.observe(action, obs)
        obs, avail = self.encode_obs(obs)
        if self.share_reward:
            global_reward = np.mean(reward)
            reward = [[global_reward]] * self.num_agents

        done = np.array([done] * self.num_agents)
        info = self._info_wrapper(info)

        return obs, obs.copy(), reward, done, info, avail

    # def seed(self, seed=None):
    #     self.env.unwrapped._env._env.config.game_engine_random_seed = seed

    def close(self):
        self.env.close()

    def _info_wrapper(self, info):
        state = self.env.unwrapped.observation()
        info.update(state[0])
        info["max_steps"] = self.max_steps
        info["active"] = np.array([state[i]["active"] for i in range(self.num_agents)])
        info["designated"] = np.array(
            [state[i]["designated"] for i in range(self.num_agents)]
        )
        info["sticky_actions"] = np.stack(
            [state[i]["sticky_actions"] for i in range(self.num_agents)]
        )
        stats = self.statsobserver.get_stats()
        stat_keys = list(stats.keys())
        stat_info = {}
        for key in stat_keys:
            stat_info[key] = stats[key]
        info["stats"] = stat_info
        info = [info.copy() for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            bad_transition = False
            if state[i]["steps_left"] == 0:
                if self.reward_config["win_reward"] != 0 and self.reward_shaping:
                    bad_transition = False
                else:
                    bad_transition = True
            info[i]["bad_transition"] = bad_transition
            info[i]["yellow"] = state[i]["left_team_yellow_card"][state[i]["active"]]
            info[i]["player_active"] = state[i]["left_team_active"][state[i]["active"]]
        return info
