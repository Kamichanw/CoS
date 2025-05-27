from functools import lru_cache
import logging
import hydra
from hydra.core.hydra_config import HydraConfig
import os
import psutil
import copy
import time
import itertools
import subprocess
from typing import Dict, List

from omegaconf import DictConfig, OmegaConf
from src.myutils.gpu import *

logger = logging.getLogger(__name__)


def flatten_dict(d, parent_key="", sep=".") -> List[str]:
    """
    Flatten a dict to the format of hydra overrides. For example, the following dict:
    ```
    {
        foo: {
            key: 1
        },
        bar: [1, 2]
    }
    ```
    will be converted to ["foo.key=1", "bar=[1,2]"].
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if v is None:
            v = "null"
        elif isinstance(v, (list, tuple)):
            v = ",".join(map(str, v))
        if isinstance(v, (dict, DictConfig)):
            items.extend(flatten_dict(v, new_key, sep=sep))
        else:
            items.append(f"{new_key}={v}")
    return items


def generate_task_configs(cfg: DictConfig) -> List[DictConfig]:

    cfg_dict = OmegaConf.to_container(cfg, resolve=False)

    list_paths = []

    def traverse(cfg_subdict, current_path, list_paths):

        if isinstance(cfg_subdict, dict):
            for key, value in cfg_subdict.items():
                path = f"{current_path}.{key}" if current_path else key
                if isinstance(value, list):
                    list_paths.append(path)
                elif isinstance(value, dict):
                    traverse(value, path, list_paths)
        elif isinstance(cfg_subdict, list):
            for idx, item in enumerate(cfg_subdict):
                path = f"{current_path}[{idx}]"
                traverse(item, path, list_paths)

    traverse(cfg_dict, "", list_paths)

    if not list_paths:
        return [cfg]

    list_values = []
    for path in list_paths:
        value = OmegaConf.select(cfg, path)
        if value is None:
            raise ValueError(f"Value at path '{path}' is None, cannot split.")
        list_values.append(value)

    combinations = list(itertools.product(*list_values))

    split_cfgs = []
    for combo in combinations:
        new_cfg = copy.deepcopy(cfg)
        for path, value in zip(list_paths, combo):
            OmegaConf.update(new_cfg, path, value, merge=False)
        split_cfgs.append(new_cfg)

    return split_cfgs


def preprocess_config(cfg: DictConfig) -> DictConfig:
    """
    Load method args from config_path/{method_name}, where method_name is from cfg.methods.
    After loading, several cfg.{method_name}.
    """
    OmegaConf.set_struct(cfg, False)

    # filter keys of args belonging to all methods
    default_keys = set(hydra.compose("default_pipeline").keys())
    new_keys = set(cfg.keys()) - default_keys
    common_keys = new_keys - set(cfg.methods)

    # ignore redundant args and only keep the args already specified in default.yaml
    config_sources = HydraConfig.get().runtime.config_sources
    config_path = next(
        item["path"] for item in config_sources if item["provider"] == "main"
    )
    default_keys = set(OmegaConf.load(os.path.join(config_path, "default.yaml")).keys())
    common_args = {k: cfg[k] for k in common_keys & default_keys}

    # re-calculate method.llm.gpu_memory_utilization
    total_memory_capacity = [65536]
    assert (
        len(total_memory_capacity) == 1
    ), "All devices should have the same capacity. Different devices with different capacities are not supported yet."
    total_memory_capacity = next(iter(total_memory_capacity))

    gpu_memory_utilization = 60000 / total_memory_capacity

    cfg.update(
        {
            method: {
                "method": OmegaConf.merge(
                    dict(
                        name=method,
                    ),
                    cfg[method] if method in cfg else {},
                ),
                **common_args,
            }
            for method in cfg.methods
        }
    )

    # common keys has been merged into method args
    # we can remove them safely
    for key in common_keys:
        del cfg[key]

    return cfg


class TaskManager:
    def __init__(self, visible_devices: List[int], timeout=60):
        self.timeout = timeout
        self.visible_devices = visible_devices

        # proc_table: pid (int) -> (allocated_devices (List[int]), requires_memory_per_device (int))
        self.proc_table: Dict[int, tuple] = {}

    def get_avail_devices(self, requires_memory: int) -> List[int]:
        # those invisible devices have 0 MiB space to use
        free_memory = [
            memory_usage if device_idx in self.visible_devices else 0
            for device_idx, memory_usage in enumerate(query_free_memory())
        ]

        # update proc_table based on current memory usage
        for pid, (allocated_devices, max_memory_usage) in self.proc_table.items():
            try:
                proc = psutil.Process(pid)
                for device_idx in allocated_devices:
                    used_memory = query_proc_used_memory(
                        proc, device_idx, include_children=True
                    )
                    if used_memory < max_memory_usage // 2:
                        free_memory[device_idx] -= max_memory_usage
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                ...

        avail_devices = [
            idx for idx, mem in enumerate(free_memory) if mem >= requires_memory
        ]

        return avail_devices

    def task_status(self) -> List[Union[int, None]]:
        """
        Get the return code of all task processes. For a task, if it ends successfully, returns 0.
        If it fails, returns a non-zero value. If it is running, returns None.
        """
        status = []
        for pid in self.proc_table:
            try:
                proc = psutil.Process(pid)
                if proc.is_running():
                    status.append(None)
                else:
                    status.append(proc.returncode)
            except psutil.NoSuchProcess:
                status.append(1)
        return status

    def dispatch(
        self, cmd, env=None, requires_memory_per_device=20000, min_devices=1, **kwargs
    ) -> List[int]:
        """
        Use cmd to open a subprocess and assign it to the appropriate devices. 
        kwargs will be passed to the subprocess. If there are enough devices, 
        the subprocess will be executed and the index of the assigned devices will be returned. 
        Otherwise, it will wait until the timeout. 
        """
        for n_tries in range(self.timeout):
            avail_devices = self.get_avail_devices(requires_memory_per_device)
            if len(avail_devices) >= min_devices:
                assigned_devices = avail_devices[:min_devices]
                env = env if env else os.environ
                proc = subprocess.Popen(
                    cmd,
                    env={
                        **env,
                        "CUDA_VISIBLE_DEVICES": ",".join(map(str, assigned_devices)),
                    },
                    text=True,
                    **kwargs,
                )

                # add the process to proc_table with its pid as the key
                for device_idx in assigned_devices:
                    self.proc_table[proc.pid] = (
                        assigned_devices,
                        requires_memory_per_device,
                    )

                return assigned_devices
            else:
                print(f"Waited for {min_devices} device(s) for {(n_tries)} mins")
                time.sleep(60)
        else:
            raise TimeoutError(
                f"Cannot find at least {min_devices} device(s). Timeout, exit..."
            )
