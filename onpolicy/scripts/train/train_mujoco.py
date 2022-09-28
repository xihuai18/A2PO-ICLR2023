#!/usr/bin/env python
import argparse
import os
from random import choices
import socket
import sys
from pathlib import Path
from pprint import pprint

import setproctitle
import torch
import yaml

from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from onpolicy.envs.ma_mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
from onpolicy.exp_utils import SacredAimExperiment
from onpolicy.exp_utils.args_utils import args_str2bool
from onpolicy.train_utils import setup_seed
"""Train script for Multi-agent Mujocos."""


def make_train_env(all_args):

    def get_env_fn(rank):

        def init_env():
            if all_args.env_name == "mujoco":
                env_args = {
                    "scenario": all_args.scenario_name,
                    "agent_conf": all_args.agent_conf,
                    "agent_obsk": all_args.agent_obsk,
                    "episode_limit": 1000
                }
                env = MujocoMulti(env_args=env_args)
            else:
                print("Can not support the " + all_args.env_name +
                      "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):

    def get_env_fn(rank):

        def init_env():
            if all_args.env_name == "mujoco":
                env_args = {
                    "scenario": all_args.scenario_name,
                    "agent_conf": all_args.agent_conf,
                    "agent_obsk": all_args.agent_obsk,
                    "episode_limit": 1000
                }
                env = MujocoMulti(env_args=env_args)
            else:
                print("Can not support the " + all_args.env_name +
                      "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


available_confs = {
    "Ant-v2": ["2x4", "2x4d", "4x2", "8x1"],
    "HalfCheetah-v2": ["2x3", "6x1", "3x2"],
    "Hopper-v2": ["3x1"],
    "Humanoid-v2": ["9|8", "17x1"],
    "HumanoidStandup-v2": ["9|8", "17x1"],
    "Walker2d-v2": ["2x3", "6x1", "3x2"],
    "Reacher-v2": ["2x1"],
    "Swimmer-v2": ["2x1"],
    "manyagent_swimmer": None,
    "manyagent_ant": None,
    "coupled_half_cheetah": ["1p1"],
}


def parse_args(args, parser: argparse.ArgumentParser):
    parser.add_argument('--scenario_name',
                        type=str,
                        default='Ant-v2',
                        choices=available_confs.keys(),
                        help="Which mujoco task to run on")
    parser.add_argument('--agent_conf', type=str, default='2x4')
    parser.add_argument('--agent_obsk', type=int, default=0)
    parser.add_argument('--mu_tanh', type=args_str2bool, default=False)
    all_args = parser.parse_args(args)

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    yaml_path = Path(
        os.path.abspath(
            os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] +
            "/../config/mujoco.yaml"))
    with open(yaml_path, "r") as yaml_file:
        map_configs = yaml.safe_load(yaml_file)
        for k, v in map_configs[all_args.scenario_name].items():
            if not isinstance(v, dict):
                assert hasattr(all_args, k), "error input in yaml config"
                setattr(all_args, k, v)
            else:
                if k == all_args.agent_conf:
                    for s_k, s_v in v:
                        assert hasattr(all_args,
                                       s_k), "error input in yaml config"
                        setattr(all_args, s_k, s_v)

    setattr(all_args, "use_" + all_args.adv, True)
    setattr(all_args, all_args.loop_order + "_loop_first", True)
    setattr(all_args, all_args.seq_strategy + "_sequence", True)

    num_agents = int(
        all_args.agent_conf.split("x")[0]) if "x" in all_args.agent_conf else (
            all_args.agent_conf.count("|") + 1)
    setattr(all_args, "num_agents", num_agents)

    if not all_args.use_agent_block:
        all_args.block_num = all_args.num_agents

    if all_args.agent_obsk < 0:
        all_args.agent_obsk = None

    pprint(all_args.__dict__)

    sanity_check(all_args)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    result_dir = (Path(
        os.path.abspath(
            os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] +
            "/../../results")) / all_args.env_name / all_args.scenario_name /
                  all_args.agent_conf / all_args.algorithm_name /
                  all_args.experiment_name)

    exp_name = f"{all_args.scenario_name}_{all_args.agent_conf}_{all_args.experiment_name}"
    code_dir = Path(
        os.path.abspath(
            os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] +
            "/../"))
    all_args.code_dir = str(code_dir)
    all_args.host = socket.gethostname()

    if not result_dir.exists():
        os.makedirs(str(result_dir))
        curr_exp = "exp1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("exp")[1])
            for folder in result_dir.iterdir()
            if str(folder.name).startswith("exp")
        ]
        if len(exst_run_nums) == 0:
            curr_exp = "exp1"
        else:
            curr_exp = "exp%i" % (max(exst_run_nums) + 1)
    exp_dir = result_dir / curr_exp
    if not exp_dir.exists():
        os.makedirs(str(exp_dir))

    all_args.exp_dir = str(exp_dir)
    print(f"the results are saved in {exp_dir}")

    logger = SacredAimExperiment(
        exp_name,
        code_dir,
        all_args.use_sacred,
        exp_dir / "logs",
        all_args.use_aim,
        all_args.aim_repo,
        not all_args.use_aim,
    )

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" +
        str(all_args.experiment_name))

    for seed in range(all_args.seed, all_args.seed + all_args.n_run):

        logger.reset()
        config = all_args.__dict__
        config.update({"argv": args})
        logger.set_config(config)
        run_name = f"{all_args.experiment_name}_seed_{all_args.seed}"
        logger.set_tag(run_name)

        setup_seed(seed)

        # env init
        envs = make_train_env(all_args)
        eval_envs = make_eval_env(all_args) if all_args.use_eval else None

        setattr(all_args, "device", device)
        setattr(all_args, "run_dir", result_dir)

        n_episode = (int(all_args.num_env_steps) // all_args.episode_length //
                     all_args.n_rollout_threads)
        setattr(all_args, "n_episode", n_episode)

        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": eval_envs,
            "num_agents": num_agents,
            "device": device,
            "run_dir": exp_dir,
            "logger": logger,
        }

        pprint(all_args.__dict__)

        # run experiments
        from onpolicy.runner.mujoco_runner import MujocoRunner as Runner

        if all_args.use_sacred:
            import sacred

            sacred_exp = logger.get_sacred_exp()

            @sacred_exp.main
            def sacred_run(_run):
                # pprint(sacred_exp.current_run.info)
                # pprint(sacred_exp.current_run.experiment_info)
                # pprint(sacred_exp.current_run.meta_info)
                config["logger"].set_sacred_run(_run)
                runner = Runner(config)
                runner.run()
                runner.logger.close()

            sacred_exp.run()
        else:
            runner = Runner(config)
            runner.run()
            runner.logger.close()

        # post process
        envs.close()
        if all_args.use_eval and eval_envs is not envs:
            eval_envs.close()

        if all_args.use_eval and eval_envs is not envs:
            eval_envs.close()


def sanity_check(all_args):
    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or
                all_args.use_naive_recurrent_policy), "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False
                and all_args.use_naive_recurrent_policy
                == False), "check recurrent policy!"
    else:
        raise NotImplementedError

    assert all_args.share_policy == False, "individual policy for mujoco"

    assert all_args.use_cum_sequence == False

    assert not all_args.print_each_agent_info or not all_args.use_sequential or all_args.agent_loop_first
    assert not all_args.use_agent_block or all_args.block_num < all_args.num_agents

    assert available_confs[
        all_args.
        scenario_name] == None or all_args.agent_conf in available_confs[
            all_args.scenario_name]

    assert (all_args.episode_length %
            all_args.data_chunk_length == 0), "chunk length requirement!"


if __name__ == "__main__":
    main(sys.argv[1:])
