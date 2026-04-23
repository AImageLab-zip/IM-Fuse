import os
import sys

import typer


def distributed_launch_active() -> bool:
    return "LOCAL_RANK" in os.environ or int(os.environ.get("WORLD_SIZE", "1")) > 1


def infer_nproc_per_node() -> int:
    import torch

    num_devices = torch.cuda.device_count()
    if num_devices <= 0:
        raise typer.BadParameter(
            "unable to infer --nproc-per-node because no CUDA devices are visible",
            param_hint="--nproc-per-node",
        )
    return num_devices


def relaunch_with_torchrun(nproc_per_node: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(nproc_per_node),
        sys.argv[0],
        *sys.argv[1:],
    ]
    raise typer.Exit(os.spawnvp(os.P_WAIT, sys.executable, command))
