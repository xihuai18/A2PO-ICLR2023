from itertools import chain
from typing import Iterable, List, Optional, Union

import numpy as np
import torch
from torch import optim

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import (R_Actor,
                                                                  R_Critic)
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.utils.util import update_linear_schedule


class Multi_Optimizer:

    def __init__(self, optimizer_list: List[optim.Optimizer]) -> None:
        self.optimizer_list = optimizer_list

    def step(self, agent_id: Optional[Union[int, Iterable]] = None) -> None:
        if agent_id is None:
            agent_id = np.arange(len(self.optimizer_list))
        elif not np.iterable(agent_id):
            agent_id = np.array([agent_id])

        for i in agent_id:
            self.optimizer_list[i].step()

    def zero_grad(self,
                  agent_id: Optional[Union[int, Iterable]] = None) -> None:
        if agent_id is None:
            agent_id = np.arange(len(self.optimizer_list))
        elif not np.iterable(agent_id):
            agent_id = np.array([agent_id])

        for i in agent_id:
            self.optimizer_list[i].zero_grad()


class R_Multi_MAPPOPolicy(R_MAPPOPolicy):
    """only support agents with the same action space and observation space

    """

    def __init__(self,
                 args,
                 obs_space,
                 cent_obs_space,
                 act_space,
                 device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.num_agents = args.num_agents

        self.actor_list = [
            R_Actor(args, self.obs_space, self.act_space, self.device)
            for _ in range(self.num_agents)
        ]
        self.critic_list = [
            R_Critic(args, self.share_obs_space, self.device)
            for _ in range(self.num_agents)
        ]

        self.actor_optimizer = Multi_Optimizer([
            torch.optim.Adam(
                self.actor_list[i].parameters(),
                lr=self.lr,
                eps=self.opti_eps,
                weight_decay=self.weight_decay,
            ) for i in range(self.num_agents)
        ])
        self.critic_optimizer = Multi_Optimizer([
            torch.optim.Adam(
                self.critic_list[i].parameters(),
                lr=self.lr,
                eps=self.opti_eps,
                weight_decay=self.weight_decay,
            ) for i in range(self.num_agents)
        ])

        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for actor in self.actor_list:
            for param in actor.parameters():
                mulValue = np.prod(param.size())
                Total_params += mulValue
                if param.requires_grad:
                    Trainable_params += mulValue
                else:
                    NonTrainable_params += mulValue
        for critic in self.critic_list:
            for param in critic.parameters():
                mulValue = np.prod(param.size())
                Total_params += mulValue
                if param.requires_grad:
                    Trainable_params += mulValue
                else:
                    NonTrainable_params += mulValue
        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')

    def actor_parameters(self,
                         agent_id: Optional[Union[int, Iterable]] = None):
        if agent_id is None:
            agent_id = np.arange(self.num_agents)
        elif not np.iterable(agent_id):
            agent_id = np.array([agent_id])

        return chain(*[self.actor_list[i].parameters() for i in agent_id])

    def critic_parameters(self,
                          agent_id: Optional[Union[int, Iterable]] = None):
        if agent_id is None:
            agent_id = np.arange(self.num_agents)
        elif not np.iterable(agent_id):
            agent_id = np.array([agent_id])

        return chain(*[self.critic_list[i].parameters() for i in agent_id])

    def lr_decay(self, episode, episodes):
        for i in range(self.num_agents):
            update_linear_schedule(self.actor_optimizer.optimizer_list[i],
                                   episode, episodes, self.lr)
            update_linear_schedule(self.critic_optimizer.optimizer_list[i],
                                   episode, episodes, self.critic_lr)

    def get_actions(self,
                    cent_obs: np.ndarray,
                    obs: np.ndarray,
                    rnn_states_actor: np.ndarray,
                    rnn_states_critic: np.ndarray,
                    masks: np.ndarray,
                    available_actions: np.ndarray = None,
                    deterministic: bool = False):
        """
        input shape (n * num_agents, *feature)
        """
        origin_shape = cent_obs.shape[:-1]
        cent_obs = cent_obs.reshape(-1, self.num_agents, cent_obs.shape[-1])
        obs = obs.reshape(-1, self.num_agents, obs.shape[-1])
        rnn_states_actor = rnn_states_actor.reshape(-1, self.num_agents, 1,
                                                    rnn_states_actor.shape[-1])
        rnn_states_critic = rnn_states_critic.reshape(
            -1, self.num_agents, 1, rnn_states_critic.shape[-1])
        masks = masks.reshape(-1, self.num_agents, masks.shape[-1])
        if available_actions is not None:
            available_actions = available_actions.reshape(
                -1, self.num_agents, available_actions.shape[-1])

        actions = []
        action_log_probs = []
        _rnn_states_actor = []
        values = []
        _rnn_states_critic = []

        for i in range(self.num_agents):
            _action, _action_log_prob, _rnn_state_actor = self.actor_list[i](
                obs[:, i], rnn_states_actor[:, i], masks[:, i],
                available_actions[:, i]
                if available_actions is not None else None, deterministic)
            _value, _rnn_state_critic = self.critic_list[i](
                cent_obs[:, i], rnn_states_critic[:, i], masks[:, i])
            actions.append(_action)
            action_log_probs.append(_action_log_prob)
            _rnn_states_actor.append(_rnn_state_actor)
            values.append(_value)
            _rnn_states_critic.append(_rnn_state_critic)
            # shape [(n, feature)]

        def _convert(x: List[torch.Tensor]):
            x = torch.stack(x, dim=1)
            x = x.reshape(*origin_shape, *x.shape[2:])
            return x

        return _convert(values), _convert(actions), _convert(
            action_log_probs), _convert(_rnn_states_actor), _convert(
                _rnn_states_critic)

    def get_values(self,
                   cent_obs: np.ndarray,
                   rnn_states_critic: np.ndarray,
                   masks: np.ndarray,
                   agent_id: Optional[Union[int, Iterable]] = None):
        if agent_id is None:
            agent_id = np.arange(self.num_agents)
        elif not np.iterable(agent_id):
            agent_id = np.array([agent_id])
        n_agents = len(agent_id)
        origin_shape = cent_obs.shape[:-1]
        cent_obs = cent_obs.reshape(-1, n_agents, cent_obs.shape[-1])
        rnn_states_critic = rnn_states_critic.reshape(
            -1, n_agents, 1, rnn_states_critic.shape[-1])
        masks = masks.reshape(-1, n_agents, masks.shape[-1])

        def _convert(x: List[torch.Tensor]):
            x = torch.stack(x, dim=1)
            x = x.reshape(*origin_shape, *x.shape[2:])
            return x

        values = []
        for i, a_i in enumerate(agent_id):
            _value, _ = self.critic_list[a_i](cent_obs[:, i],
                                              rnn_states_critic[:, i],
                                              masks[:, i])
            values.append(_value)

        return _convert(values)

    def actor_evaluate_actions(self,
                               obs: np.ndarray,
                               rnn_states_actor: np.ndarray,
                               action: np.ndarray,
                               masks: np.ndarray,
                               available_actions: np.ndarray = None,
                               active_masks: bool = False,
                               agent_id: Optional[Union[int,
                                                        Iterable]] = None):
        if agent_id is None:
            agent_id = np.arange(self.num_agents)
        elif not np.iterable(agent_id):
            agent_id = np.array([agent_id])
        n_agents = len(agent_id)

        origin_shape = obs.shape[:-1]
        obs = obs.reshape(-1, n_agents, obs.shape[-1])
        rnn_states_actor = rnn_states_actor.reshape(-1, n_agents, 1,
                                                    rnn_states_actor.shape[-1])
        action = action.reshape(-1, n_agents, action.shape[-1])
        masks = masks.reshape(-1, n_agents, masks.shape[-1])
        if available_actions is not None:
            available_actions = available_actions.reshape(
                -1, n_agents, available_actions.shape[-1])
        active_masks = active_masks.reshape(-1, n_agents, masks.shape[-1])

        action_log_probs = []
        dist_entropy = []

        for i, a_i in enumerate(agent_id):
            _action_log_prob, _dist_entropy = self.actor_list[
                a_i].evaluate_actions(
                    obs[:, i], rnn_states_actor[:, i], action[:, i],
                    masks[:, i], available_actions[:, i] if available_actions
                    is not None else None, active_masks[:, i])
            action_log_probs.append(_action_log_prob)
            dist_entropy.append(_dist_entropy)
            # shape [(n, feature)]

        def _convert(x: List[torch.Tensor]):
            x = torch.stack(x, dim=1)
            x = x.reshape(*origin_shape, *x.shape[2:])
            return x

        return _convert(action_log_probs), torch.sum(torch.stack(dist_entropy))

    def evaluate_actions(self,
                         cent_obs: np.ndarray,
                         obs: np.ndarray,
                         rnn_states_actor: np.ndarray,
                         rnn_states_critic: np.ndarray,
                         action: np.ndarray,
                         masks: np.ndarray,
                         available_actions: np.ndarray = None,
                         active_masks: bool = False,
                         agent_id: Optional[Union[int, Iterable]] = None):
        """
        input shape (n * len(agent_id), *feature)
        """
        if agent_id is None:
            agent_id = np.arange(self.num_agents)
        elif not np.iterable(agent_id):
            agent_id = np.array([agent_id])
        n_agents = len(agent_id)

        origin_shape = cent_obs.shape[:-1]
        cent_obs = cent_obs.reshape(-1, n_agents, cent_obs.shape[-1])
        obs = obs.reshape(-1, n_agents, obs.shape[-1])
        rnn_states_actor = rnn_states_actor.reshape(-1, n_agents, 1,
                                                    rnn_states_actor.shape[-1])
        rnn_states_critic = rnn_states_critic.reshape(
            -1, n_agents, 1, rnn_states_critic.shape[-1])
        action = action.reshape(-1, n_agents, action.shape[-1])
        masks = masks.reshape(-1, n_agents, masks.shape[-1])
        if available_actions is not None:
            available_actions = available_actions.reshape(
                -1, n_agents, available_actions.shape[-1])
        active_masks = active_masks.reshape(-1, n_agents, masks.shape[-1])

        action_log_probs = []
        dist_entropy = []
        values = []

        for i, a_i in enumerate(agent_id):
            _action_log_prob, _dist_entropy = self.actor_list[
                a_i].evaluate_actions(
                    obs[:, i], rnn_states_actor[:, i], action[:, i],
                    masks[:, i], available_actions[:, i] if available_actions
                    is not None else None, active_masks[:, i])
            _value, _ = self.critic_list[a_i](cent_obs[:, i],
                                              rnn_states_critic[:, i],
                                              masks[:, i])
            action_log_probs.append(_action_log_prob)
            dist_entropy.append(_dist_entropy)
            values.append(_value)
            # shape [(n, feature)]
        def _convert(x: List[torch.Tensor]):
            x = torch.stack(x, dim=1)
            x = x.reshape(*origin_shape, *x.shape[2:])
            return x

        return _convert(values), _convert(action_log_probs), torch.sum(
            torch.stack(dist_entropy))

    def act(self,
            obs: np.ndarray,
            rnn_states_actor: np.ndarray,
            masks: np.ndarray,
            available_actions: np.ndarray = None,
            deterministic: bool = False):
        origin_shape = obs.shape[:-1]
        obs = obs.reshape(-1, self.num_agents, obs.shape[-1])
        rnn_states_actor = rnn_states_actor.reshape(-1, self.num_agents, 1,
                                                    rnn_states_actor.shape[-1])
        masks = masks.reshape(-1, self.num_agents, masks.shape[-1])
        if available_actions is not None:
            available_actions = available_actions.reshape(
                -1, self.num_agents, available_actions.shape[-1])

        actions = []
        _rnn_states_actor = []

        for i in range(self.num_agents):
            _action, _, _rnn_state_actor = self.actor_list[i](
                obs[:, i], rnn_states_actor[:, i], masks[:, i],
                available_actions[:, i]
                if available_actions is not None else None, deterministic)
            actions.append(_action)
            _rnn_states_actor.append(_rnn_state_actor)

        def _convert(x: List[torch.Tensor]):
            x = torch.stack(x, dim=1)
            x = x.reshape(*origin_shape, *x.shape[2:])
            return x

        return _convert(actions), _convert(_rnn_states_actor)

    def train(self):
        for i in range(self.num_agents):
            self.critic_list[i].train()
            self.actor_list[i].train()

    def eval(self):
        for i in range(self.num_agents):
            self.critic_list[i].eval()
            self.actor_list[i].eval()
