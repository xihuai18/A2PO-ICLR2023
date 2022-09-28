from typing import Dict, List, Optional

import gym
import numpy as np
import supersuit as ss
from gym.spaces import Box
from onpolicy.envs.pettingzoo.multiagentenv import MultiAgentEnv

import pettingzoo
from pettingzoo.mpe import simple_reference_v2 as simple_reference
from pettingzoo.mpe import \
    simple_speaker_listener_v3 as simple_speaker_listener
from pettingzoo.mpe import simple_spread_v2 as simple_spread
from pettingzoo.sisl import multiwalker_v9 as multiwalker

scenario_list = [
    "multiwalker", "simple_spread", "simple_reference",
    "simple_speaker_listener"
]


def get_dim_from_act_space(act_space: gym.Space):
    if act_space.__class__.__name__ == "Discrete":
        act_dim = act_space.n
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_dim = sum(act_space.high - act_space.low + 1)
    elif act_space.__class__.__name__ == "Box":
        act_dim = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_dim = act_space.shape[0]
    else:  # agar
        act_dim = act_space[0].shape[0] + act_space[1].n
    return act_dim


def get_padded_act_space(act_space: gym.Space, max_dim: int):
    if act_space.__class__.__name__ == "Discrete":
        return gym.spaces.Discrete(max_dim)
    else:
        raise NotImplementedError


def get_unpadded_act(act_space: gym.Space, act: np.ndarray):
    if act_space.__class__.__name__ == "Discrete":
        return act % act_space.n
    else:
        raise NotImplementedError


class PettingZooEnv(MultiAgentEnv):

    def __init__(self,
                 scenario: str,
                 env_args: dict = {},
                 concat_obs_insteadof_state: bool = False) -> None:
        super().__init__()
        assert scenario in scenario_list
        self.scenario = scenario
        self.concat_obs_insteadof_state = concat_obs_insteadof_state

        self.env: pettingzoo.ParallelEnv = eval(scenario).parallel_env(
            **env_args)
        assert hasattr(self.env, "state_space") or concat_obs_insteadof_state
        self.env.reset()

        self.agent2id = {
            agent: i
            for i, agent in enumerate(self.env.possible_agents)
        }
        self.id2agent = {i: agent for agent, i in self.agent2id.items()}
        self.agents = self.env.possible_agents
        self.n = self.env.num_agents
        self.n_agents = self.env.num_agents

        self.acdims = [
            get_dim_from_act_space(self.env.action_spaces[a])
            for a in self.agents
        ]

        self.obs_size = max([
            o_s.shape[0] for _, o_s in self.env.observation_spaces.items()
        ]) + self.n_agents
        self.share_obs_size = self.get_state_size()

        self.observation_space = [
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size, ))
            for id in range(self.n)
        ]
        self.share_observation_space = [
            Box(low=float("-inf"),
                high=float("inf"),
                shape=(self.share_obs_size, )) for _ in range(self.n_agents)
        ]

        self.n_actions = max(self.acdims)

        self.act_padded = [
            True if self.acdims[a] < self.n_actions else False
            for a in range(self.n_agents)
        ]

        self.action_space = [
            self.env.action_spaces[self.id2agent[id]]
            if not self.act_padded[id] else get_padded_act_space(
                self.env.action_spaces[self.id2agent[id]], self.n_actions)
            for id in range(self.n)
        ]

        env = self.env.aec_env
        while not hasattr(env, "max_cycles"):
            env = env.env

        self.episode_limit = env.max_cycles

    def step(self, actions):
        actions_dict = {}
        for a_i in range(self.n_agents):
            if self.act_padded[a_i]:
                actions_dict[self.id2agent[a_i]] = get_unpadded_act(
                    self.env.action_spaces[self.id2agent[a_i]], actions[a_i])
            else:
                actions_dict[self.id2agent[a_i]] = actions[a_i]
        # print(actions_dict)
        for a_i, a in enumerate(self.agents):
            if self.action_space[a_i].__class__.__name__ == "Box":
                action = actions_dict[a]
                actions_dict[a] = np.clip(action, self.action_space[a_i].low,
                                          self.action_space[a_i].high)
        obs_dict, rewards, dones, infos = self.env.step(actions_dict)

        self.steps += 1

        _obs = self._preprocess_dict(obs_dict)
        _obs = self._preprocess_obs(_obs)
        obs = self._process_obs(_obs)
        rewards = self._preprocess_dict(rewards)
        rewards = [[rew] for rew in rewards]
        dones = self._preprocess_dict(dones)
        infos = self._preprocess_dict(infos)
        bad_transition = False
        if self.steps >= self.episode_limit:
            bad_transition = True
        for a_i in range(self.n_agents):
            infos[a_i]["bad_transition"] = bad_transition
        state = self.get_state(_obs)
        return obs, state, rewards, dones, infos, self.get_avail_actions()

    def reset(self):
        self.steps = 0
        obs_dict = self.env.reset()
        _obs = self._preprocess_dict(obs_dict)
        _obs = self._preprocess_obs(_obs)
        obs = self._process_obs(_obs)
        state = self.get_state(_obs)
        return obs, state, self.get_avail_actions()

    def _preprocess_obs(self, obs: List[np.ndarray]):
        out = []
        for a_i in range(self.n_agents):
            input_i = obs[a_i]
            input_i = np.concatenate([
                input_i,
                np.zeros(self.obs_size - self.n_agents - input_i.shape[-1])
            ])
            out.append(input_i)
        return out

    def _preprocess_dict(self, input_dict: Dict[str, np.ndarray]):
        input_n = []
        for a_i in range(self.n_agents):
            input_i = input_dict[self.id2agent[a_i]]
            input_n.append(input_i)
        return input_n

    def _process_obs(self, obs: List[np.ndarray]):
        obs_n = []
        for a_i in range(self.n_agents):
            obs_i = obs[a_i]
            obs_i = np.concatenate([obs_i, np.eye(self.n_agents)[a_i]])
            obs_i = (obs_i - np.mean(obs_i)) / np.std(obs_i)
            obs_n.append(obs_i)
        return obs_n

    def get_state(self, obs: List[np.ndarray] = None):
        """obs should be the list of origin obses"""
        if self.concat_obs_insteadof_state:
            assert obs is not None
            share_obs = []
            share = np.concatenate(obs, axis=-1)
            for a_i in range(self.n_agents):
                share_i = np.concatenate([share, np.eye(self.n_agents)[a_i]])
                share_i = (share_i - np.mean(share_i)) / np.std(share_i)
                share_obs.append(share_i)
        else:
            share_obs = []
            state = self.env.state()
            for a_i in range(self.n_agents):
                share_i = state
                share_i = np.concatenate([share_i, np.eye(self.n_agents)[a_i]])
                share_i = (share_i - np.mean(share_i)) / np.std(share_i)
                share_obs.append(share_i)
        return share_obs

    def get_state_size(self):
        if self.concat_obs_insteadof_state:
            return (self.obs_size -
                    self.n_agents) * self.n_agents + self.n_agents
        elif hasattr(self.env, "state_space"):
            return self.env.state_space.shape[0] + self.n_agents
        else:
            raise NotImplementedError

    def get_avail_actions(self):  # all actions are always available
        mask = np.ones(shape=(
            self.n_agents,
            self.n_actions,
        ))
        if hasattr(self, "acdims"):
            for a_i in range(self.n_agents):
                mask[a_i][self.acdims[a_i]:] = 0
        return mask

    def close(self):
        self.env.close()

    def seed(self, seed: int = None):
        self.env.reset(seed)

    def get_env_info(self):

        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.obs_size,
            "obs_spaces": self.observation_space,
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
            "action_spaces": self.action_space,
            "actions_dtype": np.float32,
            "normalise_actions": False
        }
        return env_info


if __name__ == "__main__":
    env = PettingZooEnv("multiwalker",
                        concat_obs_insteadof_state=True,
                        env_args={"shared_reward": False})
    # env = PettingZooEnv("simple_spread",
    #                     concat_obs_insteadof_state=False,
    #                     env_args={"local_ratio": 0.5})
    obs, state, avail_action = env.reset()
    for _ in range(25):
        act = []
        for a_i in range(env.n_agents):
            act.append(env.action_space[a_i].sample())
        obs, state, reward, done, info, avail_action = env.step(act)
        print(act)
        # print([o.shape for o in obs])
        # print([s.shape for s in state])
        print(reward)
    # print(done)
    # print(info)
    # print(avail_action)

    # print(env.get_env_info())
