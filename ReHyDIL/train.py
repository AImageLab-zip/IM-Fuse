# -*- coding: utf-8 -*-

import os
import sys
import torch
import argparse
import logging
from utils.stage_driver import StageConfig, run_stage
from pathlib import Path
import wandb
import numpy as np

torch.set_float32_matmul_precision('high')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s, %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_args():
    p = argparse.ArgumentParser(description="ReHyDIL stage-by-stage trainer")

    # Data and outputs
    p.add_argument("--stages", type=str, default="t1n,t2w,t2f,t1c",
                   help="Training order (comma-separated), e.g., t1n,t2w,t2f,t1c")
    p.add_argument("--current-stage",type=str,required=True)
    p.add_argument("--train_lists", type=str, default="",
                   help="Comma-separated train lists for each stage; "
                        "if a SINGLE path is given, it will be reused for all stages. "
                        "If empty, --train_fmt will be used.")
    p.add_argument("--val_lists", type=str, default="",
                   help="Comma-separated val lists for each stage; "
                        "if a SINGLE path is given, it will be reused for all stages. "
                        "If empty, --val_fmt will be used.")
    p.add_argument("--train_fmt", type=str, default="train_{mod}.list",
                   help="Template when --train_lists is empty. "
                        "If it contains {mod}, expand per-stage; otherwise reuse the same path.")
    p.add_argument("--val_fmt", type=str, default="val_{mod}.list",
                   help="Template when --val_lists is empty. "
                        "If it contains {mod}, expand per-stage; otherwise reuse the same path.")

    p.add_argument("--max_epoch", type=int, default=80)
    p.add_argument("--images_rate", type=float, default=1.0)
    p.add_argument("--base_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=4e-4)
    p.add_argument("--optim_name", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    p.add_argument("--lr_scheduler", type=str, default="warmupMultistep",
                   choices=["warmupMultistep", "warmupCosine", "autoReduce"])
    p.add_argument("--step_num_lr", type=int, default=4)

    p.add_argument("--tversky_w", type=float, default=7.0)
    p.add_argument("--imb_w", type=float, default=8.0)
    p.add_argument("--nce_weight", type=float, default=3.5)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=1.5)
    p.add_argument("--gamma", type=float, default=1.2)

    p.add_argument("--mem_size", type=int, default=2000,
                   help="Max size for balance queue / replay (stored on CPU float16)")
    p.add_argument("--p_keep", type=float, default=0.10,
                   help="Proportion of near-median-loss samples kept at the end of each stage")

    p.add_argument("--in_channels", type=int, default=4,
                   help="Input channels (set 4 if you want 4-channel inputs)")
    p.add_argument("--gpus", type=str, default="0",
                   help="CUDA_VISIBLE_DEVICES value, e.g., 0 or 0,1")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--datapath", type=Path, required=True,
                help="Root directory of your dataset")
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-epochs", type=int, default=80)
    p.add_argument("--wandb-project-name",type=str,default=None)
    return p.parse_args()


def main():
    args = parse_args()

    #os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"[Device] {device}, GPUs={os.environ.get('CUDA_VISIBLE_DEVICES','-')}")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    assert len(stages) >= 1, "Please provide at least one stage (modality)."

    train_lists = Path(__file__).parent / 'datalist' / 'train.txt'
    val_lists = Path(__file__).parent / 'datalist' / 'val.txt'


    os.makedirs(args.datapath / 'out', exist_ok=True)

    seen = []
    prev_base_dir = None
    if args.wandb_project_name is not None:
        wandb.init(
            project=args.wandb_project_name,
            name = f'training: {args.current_stage}'
        )
    for i, mod in enumerate(stages):
        base_dir = args.datapath / 'out'/ f"res-{mod}"
        prev_img_modes = seen.copy() if seen else None
        print(f'Training stage: {mod}')
        cfg = StageConfig(
            base_dir=base_dir,
            data_path=args.datapath,
            train_list=train_lists,
            val_list=val_lists,
            img_mode=mod,
            checkpoint_path=args.checkpoint_path,

            prev_img_modes=prev_img_modes,
            prev_base_dir=prev_base_dir,

            mem_size=args.mem_size,
            p_keep=args.p_keep,

            max_epoch=args.num_epochs,
            batch_size=args.batch_size,
            images_rate=args.images_rate,

            base_lr=args.base_lr * np.sqrt(args.batch_size / 64),
            weight_decay=args.weight_decay,
            optim_name=args.optim_name,

            lr_scheduler=args.lr_scheduler,
            step_num_lr=args.step_num_lr,

            tversky_w=args.tversky_w,
            imb_w=args.imb_w,
            nce_weight=args.nce_weight,

            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,

            in_channels=args.in_channels,
            num_workers_train=args.num_workers,
            num_workers_val=args.num_workers,
            device=device,
            seed=args.seed,
            wandb_project_name = args.wandb_project_name
        )

        logging.info(f"[Stage {i+1}/{len(stages)}] modality={mod}")
        if prev_img_modes:
            logging.info(f"  prev_img_modes: {prev_img_modes}")
        if prev_base_dir:
            logging.info(f"  prev_base_dir: {prev_base_dir}")

        if args.current_stage == mod:
            run_stage(cfg)
            break

        # Chain for next stage
        seen.append(mod)
        prev_base_dir = base_dir

    logging.info("\n[All Done] All stages finished successfully.")


if __name__ == "__main__":
    main()
