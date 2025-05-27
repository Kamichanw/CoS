import os
import shlex

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import subprocess
from src.myutils.pipeline import *
from dotenv import load_dotenv


def create_task(task_cfg, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    task_cfg.save_path = f"{run_dir}/result.json"
    log_path = f"{run_dir}/result.log"
    error_path = f"{run_dir}/error.log"
    command = (
        ["python", "./main_dataset.py", f"method={task_cfg.method.pop('name')}"]
        + flatten_dict(task_cfg)
        + [f"hydra.run.dir={run_dir}"]
    )
    with open(f"{run_dir}/cmd.sh", "w") as f:
        f.write(" ".join(shlex.quote(arg) for arg in command) + f" >{log_path} 2>{error_path}")
    with open("./cmds.sh", "a") as f:
        f.write(" ".join(shlex.quote(arg) for arg in command) + "\n")
    return dict(
        cmd=command,
        stdin=subprocess.DEVNULL,
        stdout=open(log_path, "w"),
        stderr=open(error_path, "w"),
    )


def print_once(id, *args, **kwarg):
    if not hasattr(print_once, "cache"):
        print_once.cache = set()
    if id not in print_once.cache:
        print_once.cache.add(id)
        print(*args, **kwarg)


@hydra.main(version_base=None, config_path="configs", config_name="default_pipeline")
def main(cfg: DictConfig):
    run_dir = HydraConfig.get().run.dir
    cfg = preprocess_config(cfg)
    task_manager = TaskManager(
        visible_devices=cfg.visible_devices, timeout=cfg.wait_devices_timeout
    )

    task_cfg_queue = [
        tasks for method in cfg.methods for tasks in generate_task_configs(cfg[method])
    ]
    task_idx, total_tasks = 0, len(task_cfg_queue)

    if os.path.exists("./cmds.sh"):
        os.remove("./cmds.sh")

    print(f"Starting running {total_tasks} tasks...")
    for task_idx, task_cfg in enumerate(task_cfg_queue):
        task_dir = os.path.join(run_dir, f"task_{task_idx+1}")
        create_task(task_cfg, run_dir=task_dir)

    # while task_idx < total_tasks:
    #     task_cfg = task_cfg_queue.pop(0)
    #     task_dir = os.path.join(run_dir, f"task_{task_idx+1}")
    #     task = create_task(task_cfg, run_dir=task_dir)
    #     try:
    #         ret = task_manager.dispatch(
    #             **task,
    #             requires_memory_per_device=cfg.requires_memory,
    #             min_devices=cfg.wait_n_devices,
    #         )
    #         print(
    #             f"Task {task_idx + 1} is assigned to devices {', '.join(map(str,ret))}"
    #         )
    #         task_idx += 1
    #     except Exception:
    #         import traceback

    #         print(f"Error occurred when running on the {task_idx+1}th task:")
    #         traceback.print_exc(file=task["stderr"])
    #         raise

    #     for idx, ret_code in enumerate(task_manager.task_status()):
    #         if ret_code is not None:
    #             if ret_code != 0:
    #                 print_once(
    #                     task_idx,
    #                     f"The {idx+1}th task exited with non-zero code, see {task_dir} for full information.",
    #                 )
    #             else:
    #                 print_once(
    #                     task_idx,
    #                     f"The {idx+1}th task ended successfully, see {task_dir} for results.",
    #                 )


if __name__ == "__main__":
    load_dotenv()
    main()
