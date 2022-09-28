from pprint import pprint
from typing import Tuple

import numpy as np
import torch

from onpolicy.utils.util import (_t2n, get_shape_from_act_space,
                                 get_shape_from_obs_space)


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _flatten_chunk(T: int, N: int, A: int, x: np.ndarray) -> np.ndarray:
    """
    A*T, N, Dim -> T, N*A, Dim -> -1, Dim
    """
    x = x.reshape(A, T, N, *x.shape[2:])
    x = x.transpose(1, 2, 0,
                    *(list(range(3, x.ndim)))).reshape(-1, *x.shape[3:])
    return x


def _split2chunk(x: np.ndarray, data_chunk_length: int,
                 num_data_chunk: int) -> np.ndarray:
    """len, thread, agent, feature -> n_c, agent * c_l, feature"""
    # thread, len, agent, feature
    x = x.transpose(1, 0, *list(range(2, x.ndim)))
    # n_c, c_l, agent, feature
    x = x.reshape(num_data_chunk, data_chunk_length, *x.shape[2:])
    # n_c, agent, c_l, feature
    x = x.transpose(0, 2, 1, *list(range(3, x.ndim)))
    # n_c, agent * c_l, feature
    x = x.reshape(num_data_chunk, -1, *x.shape[3:])
    return x


class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.num_agents = num_agents
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self._approx_trace = args.approx_trace
        self._speedup_trace = args.speedup_trace
        self._leaky_trace = args.leaky_trace
        self._leaky_alpha = args.leaky_alpha
        self._action_aggregation = args.action_aggregation

        self._share_policy = args.share_policy

        self._use_gae_trace = args.use_gae_trace
        self._use_state_IS = args.use_state_IS
        self._use_speedup_IS = args.use_speedup_IS
        self._use_trace_IS = args.use_trace_IS
        self._IS_lambda = args.IS_lambda
        self._trace_clip_param = args.trace_clip_param

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *share_obs_shape,
            ),
            dtype=np.float32,
        )
        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents,
             *obs_shape),
            dtype=np.float32,
        )

        self.rnn_states = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )
        self.returns = np.zeros_like(self.value_preds)
        if self._use_gae_trace:
            self.weighted_returns = np.zeros_like(self.value_preds)
        if self._use_state_IS:
            # \prod_{i=0}^{t-1} \frac{\pi(a_{i}|s_{i})}{\beta(a_{i}|s_{i})}
            self.prod_ratios = np.ones_like(self.value_preds)
        if self._use_trace_IS:
            self.lambda_matrix = np.ones_like(self.value_preds)

        if act_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones(
                (
                    self.episode_length + 1,
                    self.n_rollout_threads,
                    num_agents,
                    act_space.n,
                ),
                dtype=np.float32,
            )
        elif act_space.__class__.__name__ == "Box":
            self.available_actions = np.ones(
                (
                    self.episode_length + 1,
                    self.n_rollout_threads,
                    num_agents,
                    act_space.shape[0],
                ),
                dtype=np.float32,
            )
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents,
             act_shape),
            dtype=np.float32,
        )
        if act_space.__class__.__name__ in [
                "Discrete", "MultiBinary", "MultiDiscrete"
        ]:
            self.action_log_probs = np.zeros(
                (self.episode_length, self.n_rollout_threads, num_agents,
                 act_shape),
                dtype=np.float32,
            )
        elif act_space.__class__.__name__ == "Box":
            self.action_log_probs = np.zeros(
                (self.episode_length, self.n_rollout_threads, num_agents, 1),
                dtype=np.float32,
            )
        else:
            raise NotImplementedError("Mixed action not supported!")

        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )

        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        bad_masks=None,
        active_masks=None,
        available_actions=None,
    ):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()
        if self._use_state_IS:
            # the len-th data, prod from 0 to len - 1
            self.prod_ratios[0] = self.prod_ratios[-1].copy()
        if self._use_trace_IS:
            self.lambda_matrix[0] = self.lambda_matrix[-1].copy()

    @torch.no_grad()
    def some_agent_compute_ratios_from_current_policy(
        self,
        policy,
        agent_ids: np.array,
        agent_mask: np.array,
    ) -> Tuple[np.ndarray]:
        """
        return: (ratios, joint_ratios), shape (len, thread, block, feature)
        """
        agent_ids = agent_ids.reshape(-1)
        block_size = agent_ids.shape[0]
        assert agent_ids.shape[0] <= self.num_agents, "agent_ids error!"
        assert agent_mask.shape[0] == block_size, "agent mask error!"

        # ratios of all agents should be computed
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        rnn_states = self.rnn_states[:-1].reshape(-1,
                                                  *self.rnn_states.shape[3:])
        actions = self.actions.reshape(-1, *self.actions.shape[3:])
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[3:])
        available_actions = self.available_actions[:-1].reshape(
            -1, *self.available_actions.shape[3:]
        ) if self.available_actions is not None else None
        active_masks = self.active_masks[:-1].reshape(
            -1, *self.active_masks.shape[3:])

        if self._share_policy:
            cur_action_log_probs, _ = policy.actor.evaluate_actions(
                obs, rnn_states, actions, masks, available_actions,
                active_masks)
        else:
            cur_action_log_probs, _ = policy.actor_evaluate_actions(
                obs, rnn_states, actions, masks, available_actions,
                active_masks)

        cur_action_log_probs = cur_action_log_probs.reshape(
            *self.action_log_probs.shape)
        cur_action_log_probs = (cur_action_log_probs.detach().cpu()
                                )  # size: len,thread,agent,*

        ratios = torch.exp(cur_action_log_probs - torch.from_numpy(
            self.action_log_probs))  # shape: len * thread * agent * feature
        ratios = getattr(torch, self._action_aggregation)(ratios,
                                                          dim=-1,
                                                          keepdim=True)
        joint_ratios = ratios.unsqueeze(2)
        # shape: len * thread * 1 * agent * feature
        joint_ratios = torch.repeat_interleave(joint_ratios, block_size, dim=2)
        # shape: len * thread * block_size * agent * feature

        agent_mask = torch.from_numpy(agent_mask)
        agent_mask = agent_mask.reshape(
            block_size, self.num_agents,
            *([1] *
              (joint_ratios.ndim - 4)))  # shape: block_size * agent * feature
        joint_ratios[..., agent_mask == 0] = 1.0
        joint_ratios = torch.prod(joint_ratios, dim=3)

        ratios = ratios[:, :, agent_ids]

        return ratios.cpu().numpy(), joint_ratios.cpu().numpy()

    @torch.no_grad()
    def compute_ratios_from_current_policy(
        self,
        policy,
        agent_mask: np.array,
    ) -> Tuple[np.ndarray]:
        """return: ratios, joint_ratios"""
        return self.some_agent_compute_ratios_from_current_policy(
            policy, np.arange(self.num_agents), agent_mask)

    @torch.no_grad()
    def some_agent_compute_prod_ratios_from_current_policy(
        self,
        policy,
        agent_ids: np.array,
        agent_mask: np.array,
    ) -> Tuple[np.ndarray]:
        """
        \operatorname{prod\_ratio}(s_t) = \prod_{i=0}^{t-1} \frac{\pi(a_{i}|s_{i})}{\beta(a_{i}|s_{i})}

        agent_ids:
            the identifiers of the agents the gae-trace is computed for, with size `block_size`
        agent_mask:
            block_size * num_agents, a row is an one-hot array, indicating which agents should be used for off-policy correction.
            block_num is the size of the agents' block
        return: ratios
        """
        agent_ids = agent_ids.reshape(-1)
        block_size = agent_ids.shape[0]
        assert agent_ids.shape[0] <= self.num_agents, "agent_ids error!"
        assert agent_mask.shape[0] == block_size, "agent mask error!"

        _ratios, _joint_ratios = self.some_agent_compute_ratios_from_current_policy(
            policy, agent_ids, agent_mask)

        original_ratios = _ratios
        original_joint_ratios = _joint_ratios

        ratios = _joint_ratios

        if not self._use_speedup_IS:
            ratios = np.minimum(ratios, self._trace_clip_param)
        # shape: len * thread * block_size * feature

        # shape: thread * block_size * feature
        for step in range(ratios.shape[0]):  # len
            self.prod_ratios[step][:, agent_ids] = np.where(
                self.masks[step][:, agent_ids] == 0,
                1.0,
                self.prod_ratios[step][:, agent_ids],
            )
            if self._use_trace_IS:
                self.lambda_matrix[step][:, agent_ids] = np.where(
                    self.masks[step][:, agent_ids] == 0,
                    1.0,
                    self.lambda_matrix[step][:, agent_ids],
                )
            if self._use_speedup_IS:
                if self._use_trace_IS:
                    ratios[step] = np.minimum(
                        ((1 - self.lambda_matrix[step][:, agent_ids] *
                          self._IS_lambda) / (1 - self._IS_lambda)) /
                        self.prod_ratios[step][:, agent_ids],
                        ratios[step],
                    )
                else:
                    ratios[step] = np.minimum(
                        1.0 / self.prod_ratios[step][:, agent_ids],
                        ratios[step])
            if self._use_trace_IS:
                self.prod_ratios[step + 1][:, agent_ids] = (
                    self.prod_ratios[step][:, agent_ids] +
                    self.lambda_matrix[step] * self._IS_lambda) * ratios[step]
            else:
                self.prod_ratios[step + 1][:, agent_ids] = (
                    self.prod_ratios[step][:, agent_ids]) * ratios[step]
            if self._use_trace_IS:
                self.lambda_matrix[step + 1][:, agent_ids] = (
                    self.lambda_matrix[step][:, agent_ids] * self._IS_lambda)
        if self._use_trace_IS:
            self.prod_ratios[:, :, agent_ids] = (
                self.prod_ratios[:, :, agent_ids] * (1 - self._IS_lambda) /
                (1 - self.lambda_matrix[:, :, agent_ids] * self._IS_lambda))

        return original_ratios, original_joint_ratios

    @torch.no_grad()
    def compute_prod_ratios_from_current_policy(
        self,
        policy,
        agent_mask: np.array,
    ) -> Tuple[np.ndarray]:
        """
        agent_mask:
            num_agents * num_agents, a row is an one-hot array, indicating which agents should be used for off-policy correction
        return: ratios
        """
        return self.some_agent_compute_prod_ratios_from_current_policy(
            policy, np.arange(self.num_agents), agent_mask)

    @torch.no_grad()
    def some_agent_compute_returns_from_current_policy(
        self,
        policy,
        agent_ids: np.array,
        agent_mask: np.array,
        value_normalizer: torch.nn.Module = None,
    ) -> Tuple[np.ndarray]:
        """
        agent_ids:
            the identifiers of the agents the gae-trace is computed for, with size `block_size`
        agent_mask:
            block_size * num_agents, a row is an one-hot array, indicating which agents should be used for off-policy correction.
            block_num is the size of the agents' block

        this function computes gae-trace and return ratios
        """
        # NOTE: the data in the buffer has shape: len, thread, agent, feature
        # value is used only for agent agnet_id
        agent_ids = agent_ids.reshape(-1)
        block_size = agent_ids.shape[0]
        assert agent_ids.shape[0] <= self.num_agents, "agent_ids error!"
        assert agent_mask.shape[0] == block_size, "agent mask error!"
        assert self._use_gae_trace, "gae trace error!"
        if self._share_policy:
            next_value = policy.get_values(
                np.concatenate(self.share_obs[-1][:, agent_ids]),
                np.concatenate(self.rnn_states_critic[-1][:, agent_ids]),
                np.concatenate(self.masks[-1][:, agent_ids]),
            )
        else:
            next_value = policy.get_values(
                np.concatenate(self.share_obs[-1][:, agent_ids]),
                np.concatenate(self.rnn_states_critic[-1][:, agent_ids]),
                np.concatenate(self.masks[-1][:, agent_ids]), agent_ids)

        # shape: thread * block_size, feature
        next_value = np.array(
            np.split(_t2n(next_value), self.n_rollout_threads))
        # shape: thread, block_size, feature
        # MARK: may be numpy's bug
        self.value_preds[-1][:, agent_ids] = next_value

        _ratios, _joint_ratios = self.some_agent_compute_ratios_from_current_policy(
            policy, agent_ids, agent_mask)

        original_ratios = _ratios
        original_joint_ratios = _joint_ratios

        ratios = _joint_ratios

        ratios = torch.from_numpy(ratios)
        if not self._speedup_trace:
            if not self._leaky_trace:
                ratios = torch.minimum(ratios,
                                       torch.tensor([self._trace_clip_param
                                                     ])) * self.gae_lambda
            else:
                ratios = (self._leaky_alpha * torch.minimum(
                    ratios, torch.tensor([self._trace_clip_param])) +
                          (1 - self._leaky_alpha) * ratios) * self.gae_lambda
        # shape: len * thread * block_size * feature

        episode_length = ratios.shape[0]
        ones_upper = torch.triu(torch.ones((episode_length, episode_length)),
                                0)
        rates_lower = torch.tril(
            torch.ones((episode_length, episode_length)) * self.gae_lambda *
            self.gamma,
            -1,
        )
        rates = torch.tril(torch.cumprod(rates_lower + ones_upper, dim=0), 0)

        ones_upper = ones_upper.reshape(
            *([1] * (ratios.ndim - 1)),
            *ones_upper.shape)  # shape: 1 * 1 * 1 * len * len
        rates = rates.reshape(*([1] * (ratios.ndim - 1)),
                              *rates.shape)  # shape: 1 * 1 * 1 * len * len

        ratios_lower = torch.repeat_interleave(
            torch.unsqueeze(ratios, 1), episode_length,
            dim=1)  # shape: len * len * thread * block_size * X
        ratios_lower = ratios_lower.permute(
            *list(range(2, ratios_lower.ndim)), 0,
            1)  # shape: thread * block_size * X * len * len
        ratios_lower = torch.tril(ratios_lower, -1)
        if not self._speedup_trace:
            ratios = torch.cumprod(ratios_lower + ones_upper, dim=-2)
        else:
            ratios = ratios_lower + ones_upper
            for l_i in range(1, episode_length):
                ratios[..., l_i, :] = torch.minimum(
                    1 / ratios[..., l_i - 1, :],
                    ratios[..., l_i, :]) * ratios[..., l_i - 1, :]
        ratios = torch.tril(ratios, 0)

        if self._use_popart or self._use_valuenorm:
            delta = (self.rewards[:, :, agent_ids] +
                     self.gamma * value_normalizer.denormalize(
                         self.value_preds[1:, :, agent_ids], agent_ids) *
                     self.masks[1:, :, agent_ids] -
                     value_normalizer.denormalize(
                         self.value_preds[:-1, :, agent_ids], agent_ids))
        else:
            delta = (self.rewards[:, :, agent_ids] +
                     self.gamma * self.value_preds[1:, :, agent_ids] *
                     self.masks[1:, :, agent_ids] -
                     self.value_preds[:-1, :, agent_ids])
        # shape: len, thread, 1, X

        masks_lower = torch.repeat_interleave(
            torch.unsqueeze(torch.from_numpy(self.masks[:-1, :, agent_ids]),
                            1),
            episode_length,
            dim=1,
        )  # shape: len * len * thread * block_size * X
        masks_lower = masks_lower.permute(*list(range(2, masks_lower.ndim)), 0,
                                          1)
        # shape: thread * block_size * X * len * len
        masks_lower = torch.tril(masks_lower, -1)
        masks = torch.tril(torch.cumprod(masks_lower + ones_upper, dim=-2), 0)

        delta = torch.unsqueeze(torch.from_numpy(delta), 1)
        # shape: len, 1, thread, block_size, X
        delta = delta.permute(*list(range(2, delta.ndim)), 0, 1)
        # shape: thread, block_size, X, len, 1

        gae_trace = rates * ratios * delta * masks
        # shape: thread * block_size * X * len * len
        if self._use_proper_time_limits:
            bad_masks_lower = torch.repeat_interleave(
                torch.unsqueeze(
                    torch.from_numpy(self.bad_masks[1:, :, agent_ids]), 1),
                episode_length,
                dim=1,
            )  # shape: len * len * thread * block_size * X
            bad_masks_lower = bad_masks_lower.permute(
                *list(range(2, bad_masks_lower.ndim)), 0, 1)
            bad_masks_lower = torch.tril(bad_masks_lower, 0)
            ones_upper = torch.triu(
                torch.ones((episode_length, episode_length)), 1)
            ones_upper = ones_upper.reshape(
                *([1] * (bad_masks_lower.ndim - 2)),
                *ones_upper.shape)  # shape: 1 * 1 * 1 * len * len
            bad_masks = torch.tril(
                torch.cumprod(bad_masks_lower + ones_upper, dim=-2), 0)
            # if torch.any(
            #         torch.cumprod(bad_masks_lower +
            #                       ones_upper, dim=-2) == 0.0):
            #     print("bad_masks1")
            # if np.any(self.bad_masks[1:, :, agent_ids]==0.0):
            #     print("bad_masks2")
            gae_trace = gae_trace * bad_masks

        gae_trace = torch.sum(gae_trace, dim=-2)
        # shape: thread * block_size * X * len

        gae_trace = gae_trace.permute(gae_trace.ndim - 1,
                                      *list(range(gae_trace.ndim - 1)))
        # shape:
        # len * thread * block_size * X

        # assert gae_trace.cpu().numpy().shape == self.value_preds[:-1].shape, "gae trace shape"
        if self._use_popart or self._use_valuenorm:
            self.weighted_returns[:-1, :, agent_ids] = gae_trace.cpu().numpy(
            ) + value_normalizer.denormalize(
                self.value_preds[:-1, :, agent_ids], agent_ids)
        else:
            self.weighted_returns[:-1, :, agent_ids] = (
                gae_trace.cpu().numpy() + self.value_preds[:-1, :, agent_ids])

        if self._approx_trace:
            # print("copy weighted_returns to returns")
            np.copyto(self.returns[:-1, :, agent_ids],
                      self.weighted_returns[:-1, :, agent_ids])

        return original_ratios, original_joint_ratios

    @torch.no_grad()
    def compute_returns_from_current_policy(
            self,
            policy,
            agent_mask: np.array,
            value_normalizer: torch.nn.Module = None) -> Tuple[np.ndarray]:
        """
        agent_mask:
            num_agents * num_agents, a row is an one-hot array, indicating which agents should be used for off-policy correction
        """
        return self.some_agent_compute_returns_from_current_policy(
            policy, np.arange(self.num_agents), agent_mask, value_normalizer)

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        if self._use_proper_time_limits:
            if self._use_gae or self._use_gae_trace:
                self.value_preds[-1] = next_value
                gae = 0
                if self._use_popart or self._use_valuenorm:
                    denormalized_value_preds = value_normalizer.denormalize(
                        self.value_preds)
                # if np.any(self.bad_masks[1:]==0.0):
                #     print("bad_masks3")
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = (
                            self.rewards[step] +
                            self.gamma * denormalized_value_preds[step + 1] *
                            self.masks[step + 1] -
                            denormalized_value_preds[step])
                        gae = (delta + self.gamma * self.gae_lambda * gae *
                               self.masks[step + 1])
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[
                            step] = gae + denormalized_value_preds[step]
                    else:
                        delta = (self.rewards[step] +
                                 self.gamma * self.value_preds[step + 1] *
                                 self.masks[step + 1] - self.value_preds[step])
                        gae = (delta + self.gamma * self.gae_lambda *
                               self.masks[step + 1] * gae)
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                if self._use_popart or self._use_valuenorm:
                    denormalized_value_preds = value_normalizer.denormalize(
                        self.value_preds)
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma *
                            self.masks[step + 1] + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (1 - self.bad_masks[
                            step + 1]) * denormalized_value_preds[step]
                    else:
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma *
                            self.masks[step + 1] + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (1 - self.bad_masks[
                            step + 1]) * self.value_preds[step]
        else:
            if self._use_gae or self._use_gae_trace:
                self.value_preds[-1] = next_value
                gae = 0
                if self._use_popart or self._use_valuenorm:
                    denormalized_value_preds = value_normalizer.denormalize(
                        self.value_preds)
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = (
                            self.rewards[step] +
                            self.gamma * denormalized_value_preds[step + 1] *
                            self.masks[step + 1] -
                            denormalized_value_preds[step])
                        gae = (delta + self.gamma * self.gae_lambda *
                               self.masks[step + 1] * gae)
                        self.returns[
                            step] = gae + denormalized_value_preds[step]
                    else:
                        delta = (self.rewards[step] +
                                 self.gamma * self.value_preds[step + 1] *
                                 self.masks[step + 1] - self.value_preds[step])
                        gae = (delta + self.gamma * self.gae_lambda *
                               self.masks[step + 1] * gae)
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (self.returns[step + 1] * self.gamma *
                                          self.masks[step + 1] +
                                          self.rewards[step])
        # MARK: deepcopy, for the first update
        if self._use_gae_trace:
            np.copyto(self.weighted_returns, self.returns)

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        :return with size mini_batch_size * agent, feature
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        each_agent_batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert each_agent_batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(
                    n_rollout_threads,
                    episode_length,
                    num_agents,
                    n_rollout_threads * episode_length * num_agents,
                    num_mini_batch,
                ))
            each_agent_mini_batch_size = each_agent_batch_size // num_mini_batch
        else:
            each_agent_mini_batch_size = mini_batch_size

        rand = torch.randperm(each_agent_batch_size).numpy()
        sampler = [
            rand[i * each_agent_mini_batch_size:(i + 1) *
                 each_agent_mini_batch_size] for i in range(num_mini_batch)
        ]

        # shape: len*thread, agent, feature
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1,
                                                  *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(
            -1, *self.rnn_states_critic.shape[2:])
        actions = self.actions.reshape(-1, self.num_agents,
                                       self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(
                -1, self.num_agents, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, self.num_agents, 1)
        returns = self.returns[:-1].reshape(-1, self.num_agents, 1)
        masks = self.masks[:-1].reshape(-1, self.num_agents, 1)
        active_masks = self.active_masks[:-1].reshape(-1, self.num_agents, 1)
        action_log_probs = self.action_log_probs.reshape(
            -1, self.num_agents, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, self.num_agents, 1)

        for indices in sampler:
            # shape: each_agent_mini_batch_size * self.num_agent, feature
            share_obs_batch = share_obs[indices].reshape(
                -1, *share_obs.shape[2:])
            obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            rnn_states_batch = rnn_states[indices].reshape(
                -1, *rnn_states.shape[2:])
            rnn_states_critic_batch = rnn_states_critic[indices].reshape(
                -1, *rnn_states_critic.shape[2:])
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices].reshape(
                    -1, *available_actions.shape[2:])
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices].reshape(
                -1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(
                -1, *active_masks.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(
                -1, *action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(
                    -1, *advantages[indices].shape[2:])

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    # MARK: should never be used
    def naive_recurrent_generator(self, advantages, num_mini_batch):
        """
        Yield training data for non-chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * num_agents
        assert n_rollout_threads * num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_agents,
                                            num_mini_batch))
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()

        share_obs = self.share_obs.reshape(-1, batch_size,
                                           *self.share_obs.shape[3:])
        obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])
        rnn_states = self.rnn_states.reshape(-1, batch_size,
                                             *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic.reshape(
            -1, batch_size, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(
                -1, batch_size, self.available_actions.shape[-1])
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(
            -1, batch_size, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, batch_size, 1)

        for start_ind in range(0, batch_size, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(share_obs[:-1, ind])
                obs_batch.append(obs[:-1, ind])
                rnn_states_batch.append(rnn_states[0:1, ind])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, ind])
                actions_batch.append(actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size [(T,-1)] -> (T, N, -1)
            share_obs_batch = np.stack(share_obs_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch,
                                                  1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, dim) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch).reshape(
                N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(
                rnn_states_critic_batch).reshape(
                    N, *self.rnn_states_critic.shape[3:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N,
                                                   available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N,
                                                  old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, num_mini_batch,
                            data_chunk_length):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN.
        :return with size (c_l*n_c*agent, feature_size)
        """
        # len: episode_length
        # thread: num_rollout_threads
        # agent: num_agents
        # c_l: data_chunk_length
        # n_c: num of data_chunk per agent

        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        each_agent_batch_size = n_rollout_threads * episode_length
        each_agent_num_data_chunk = each_agent_batch_size // data_chunk_length
        each_agent_mini_num_data_chunk = each_agent_num_data_chunk // num_mini_batch

        rand = torch.randperm(each_agent_num_data_chunk).numpy()
        sampler = [
            rand[i * each_agent_mini_num_data_chunk:(i + 1) *
                 each_agent_mini_num_data_chunk] for i in range(num_mini_batch)
        ]

        # shape: n_c, agent * c_l, feature
        share_obs = _split2chunk(self.share_obs[:-1], data_chunk_length,
                                 each_agent_num_data_chunk)
        obs = _split2chunk(self.obs[:-1], data_chunk_length,
                           each_agent_num_data_chunk)

        actions = _split2chunk(self.actions, data_chunk_length,
                               each_agent_num_data_chunk)
        action_log_probs = _split2chunk(self.action_log_probs,
                                        data_chunk_length,
                                        each_agent_num_data_chunk)
        advantages = _split2chunk(advantages, data_chunk_length,
                                  each_agent_num_data_chunk)
        value_preds = _split2chunk(self.value_preds[:-1], data_chunk_length,
                                   each_agent_num_data_chunk)
        returns = _split2chunk(self.returns[:-1], data_chunk_length,
                               each_agent_num_data_chunk)
        masks = _split2chunk(self.masks[:-1], data_chunk_length,
                             each_agent_num_data_chunk)
        active_masks = _split2chunk(self.active_masks[:-1], data_chunk_length,
                                    each_agent_num_data_chunk)

        rnn_states = _split2chunk(self.rnn_states[:-1], data_chunk_length,
                                  each_agent_num_data_chunk)
        rnn_states_critic = _split2chunk(self.rnn_states_critic[:-1],
                                         data_chunk_length,
                                         each_agent_num_data_chunk)

        if self.available_actions is not None:
            available_actions = _split2chunk(
                self.available_actions[:-1],
                data_chunk_length,
                each_agent_num_data_chunk,
            )

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:

                ind = index
                # append (shape: agent*c_l, feature)
                share_obs_batch.append(share_obs[ind])
                obs_batch.append(obs[ind])
                actions_batch.append(actions[ind])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind])
                value_preds_batch.append(value_preds[ind])
                return_batch.append(returns[ind])
                masks_batch.append(masks[ind])
                active_masks_batch.append(active_masks[ind])
                old_action_log_probs_batch.append(action_log_probs[ind])
                adv_targ.append(advantages[ind])
                # append (shape: agent, feature)
                r_s = rnn_states[ind]
                r_s = r_s.reshape(self.num_agents, -1, *r_s.shape[1:])[:, 0]
                rnn_states_batch.append(r_s)
                r_s_c = rnn_states_critic[ind]
                r_s_c = r_s_c.reshape(self.num_agents, -1, *r_s_c.shape[1:])[:,
                                                                             0]
                rnn_states_critic_batch.append(r_s_c)

            N = each_agent_mini_num_data_chunk

            # from rnn.py, it requires the data has shape (T, M, ...), M is the shape[0] of rnn_state, i.e, N * agent -> (c_l, N, agent, Dim)
            # These are all from_numpys of size (agent*c_l, N, Dim)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch,
                                                   axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch,
                                                  axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, agent, feature) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(
                N * self.num_agents, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(
                rnn_states_critic_batch).reshape(
                    N * self.num_agents, *self.rnn_states_critic.shape[3:])

            # Flatten the (agent*c_l, N, ...) from_numpys to (c_l*N*agent, ...)
            share_obs_batch = _flatten_chunk(data_chunk_length, N,
                                             self.num_agents, share_obs_batch)
            obs_batch = _flatten_chunk(data_chunk_length, N, self.num_agents,
                                       obs_batch)
            actions_batch = _flatten_chunk(data_chunk_length, N,
                                           self.num_agents, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten_chunk(
                    data_chunk_length, N, self.num_agents,
                    available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten_chunk(data_chunk_length, N,
                                               self.num_agents,
                                               value_preds_batch)
            return_batch = _flatten_chunk(data_chunk_length, N,
                                          self.num_agents, return_batch)
            masks_batch = _flatten_chunk(data_chunk_length, N, self.num_agents,
                                         masks_batch)
            active_masks_batch = _flatten_chunk(data_chunk_length, N,
                                                self.num_agents,
                                                active_masks_batch)
            old_action_log_probs_batch = _flatten_chunk(
                data_chunk_length, N, self.num_agents,
                old_action_log_probs_batch)
            adv_targ = _flatten_chunk(data_chunk_length, N, self.num_agents,
                                      adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch
