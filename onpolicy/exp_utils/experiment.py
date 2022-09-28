import os
import socket
from pathlib import Path
from typing import Any, Dict

from tensorboardX import SummaryWriter


class SacredAimExperiment(object):
    """
    An experiment manager that integrates
    1. Sacred for
        a) configuration saving
        b) sourcecode saving
        c) evaluation results logging
        d) other reproducibility functions, such as seed saving
    2. Aim for
        a) training curves comparsion
        b) hyper-parameters comparision

    Usage:
    ```python
        # initialization
        exp = SacredAimExperiment()
        # set attribute
        exp.set_config()
        exp.set_tag()
        # set sacred run
        exp.set_sacred_run()
    ```
    """

    def __init__(
        self,
        exp_name: str,
        base_dir: str = "./exp_logs",
        use_sacred: bool = True,
        log_path: str = None,
        use_aim: bool = False,
        aim_repo_path: str = None,
        use_tb: bool = False,
    ):
        """
        Args:
            exp_name: experiment name for recognization
            base_dir: the scope (directory) of the source code
        """
        assert (use_sacred or use_aim
                or use_tb), "you must use at least one experiment manager"

        assert not (use_aim and use_tb), "only one vis manager is enough"

        self.exp_name = exp_name
        self.base_dir = Path(base_dir)
        self.log_path = Path(log_path)

        self.use_sacred = use_sacred

        self.use_aim = use_aim
        self.aim_repo_path = aim_repo_path

        self.use_tb = use_tb

        if self.use_sacred:
            import sacred
            from sacred.observers import FileStorageObserver

            self.sacred_exp = sacred.Experiment(
                self.exp_name,
                base_dir=self.base_dir,
                save_git_info=False,
            )
            self.sacred_exp.observers.append(FileStorageObserver(
                self.log_path))
            self.sacred_run: sacred.run.Run = None
            for path, _, file_list in os.walk(self.base_dir):
                for file_name in file_list:
                    if os.path.splitext(file_name)[1] == ".py":
                        # print(file_name)
                        file_path = os.path.join(path, file_name)
                        self.sacred_exp.add_source_file(file_path)

        if self.use_aim:
            self.aim_run = None

        if self.use_tb:
            self.tb_writer = SummaryWriter(self.log_path)

    def set_sacred_run(self, run: "sacred.run.Run"):
        self.sacred_run = run

    def get_sacred_exp(self):
        return self.sacred_exp

    def get_sacred_run(self):
        return self.sacred_run

    def get_aim_run(self):
        return self.aim_run

    def get_tb_writer(self):
        return self.tb_writer

    def set_config(self, config: dict):
        if self.use_sacred:
            self.sacred_exp.add_config(config)
        if self.use_aim:
            self.aim_run["hparams"] = config

    def set_tag(self, tag: str):
        if self.use_aim:
            self.aim_run.add_tag(tag)

    def reset(self):
        if self.use_aim:
            import aim

            self.aim_run = aim.Run(
                repo=self.aim_repo_path,
                experiment=self.exp_name,
                log_system_params=False,
            )

    def log_stat(self,
                 name: str,
                 value: Any,
                 step: int,
                 eval_stat: bool = False):
        # print(name, value)
        if self.use_aim:
            self.aim_run.track(value, name, step=step)
        else:
            self.tb_writer.add_scalar(name, value, step)
        if self.use_sacred and eval_stat:
            self.sacred_run.log_scalar(name, value, step)

    def log_stat_dict(self,
                      stat_dict: Dict[str, Any],
                      step: int,
                      eval_stat: bool = False):
        for key, value in stat_dict.items():
            self.log_stat(key, value, step, eval_stat)

    def close(self):
        if self.use_aim:
            self.aim_run.close()
        if self.use_tb:
            self.tb_writer.export_scalars_to_json(self.log_path /
                                                  "tb_results.json")
            self.tb_writer.close()
