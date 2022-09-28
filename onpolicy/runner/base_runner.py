import os
from typing import Optional

import numpy as np
import torch

from onpolicy.exp_utils import SacredAimExperiment
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.util import _t2n


class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """

    def __init__(self, config):

        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]
        if config.__contains__("render_envs"):
            self.render_envs = config["render_envs"]

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size

        self.share_policy = self.all_args.share_policy

        self.use_aim = self.all_args.use_aim
        self.use_sacred = self.all_args.use_sacred
        self.use_tb = not self.use_aim
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        # gae-trace
        self.use_gae_trace = self.all_args.use_gae_trace
        # sequential
        self.use_sequential = self.all_args.use_sequential

        self.logger: SacredAimExperiment = config["logger"]
        self.all_args.logger = self.logger
        self.run_dir = config["run_dir"]
        self.save_dir = str(self.run_dir / "models")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.all_args.share_policy:
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        else:
            from onpolicy.algorithms.r_mappo.algorithm.rMultiMAPPOPolicy import R_Multi_MAPPOPolicy as Policy

        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo

        share_observation_space = (self.envs.share_observation_space[0]
                                   if self.use_centralized_V else
                                   self.envs.observation_space[0])

        # policy network
        self.policy = Policy(
            self.all_args,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0],
            device=self.device,
        )

        # print(self.policy)

        print("action shape: ", self.envs.action_space[0].shape)
        print(self.envs.action_space[0])
        print(self.envs.observation_space[0])
        print(
            "obs shape: ", self.envs.observation_space[0].shape if hasattr(
                self.envs.observation_space[0], "shape") else len(
                    self.envs.observation_space[0]))
        print(
            "share obs shape", share_observation_space.shape if hasattr(
                share_observation_space, "shape") else
            len(share_observation_space))

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args,
                                 self.policy,
                                 device=self.device)

        # buffer
        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0],
        )

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_values = np.array(
            np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self, episode: Optional[int] = None):
        """Train policies with data in buffer."""
        self.trainer.prep_training()
        if self.use_sequential:
            train_infos = self.trainer.sequential_train(self.buffer,
                                                        episode=episode)
        else:
            train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        if self.share_policy:
            policy_actor = self.policy.actor
            torch.save(policy_actor.state_dict(),
                       str(self.save_dir) + "/actor.pt")
            policy_critic = self.policy.critic
            torch.save(policy_critic.state_dict(),
                       str(self.save_dir) + "/critic.pt")
        else:
            for i in range(self.num_agents):
                policy_actor = self.policy.actor_list[i]
                torch.save(policy_actor.state_dict(),
                           str(self.save_dir) + "/actor_{}.pt".format(i))
                policy_critic = self.policy.critic_list[i]
                torch.save(policy_critic.state_dict(),
                           str(self.save_dir) + "/critic_{}.pt".format(i))

    def restore(self):
        """Restore policy's networks from a saved model."""
        if self.share_policy:
            policy_actor_state_dict = torch.load(
                str(self.model_dir) + "/actor.pt")
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(
                    str(self.model_dir) + "/critic.pt")
                self.policy.critic.load_state_dict(policy_critic_state_dict)
        else:
            for i in range(self.num_agents):
                policy_actor_state_dict = torch.load(
                    str(self.model_dir) + "/actor_{}.pt".format(i))
                self.policy.actor_list[i].load_state_dict(
                    policy_actor_state_dict)
                if not self.all_args.use_render:
                    policy_critic_state_dict = torch.load(
                        str(self.model_dir) + "/critic_{}.pt".format(i))
                    self.policy.critic_list[i].load_state_dict(
                        policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        self.logger.log_stat_dict(train_infos, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if np.iterable(v):
                self.logger.log_stat(k,
                                     np.mean(v),
                                     total_num_steps,
                                     eval_stat=True)
            else:
                self.logger.log_stat(k, v, total_num_steps, eval_stat=True)
