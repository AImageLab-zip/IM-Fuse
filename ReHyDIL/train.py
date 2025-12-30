# -*- coding: utf-8 -*-

import os
import sys
import torch
import argparse
import logging
from utils.stage_driver import StageConfig, run_stage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s, %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_args():
    p = argparse.ArgumentParser(description="ReHyDIL stage-by-stage trainer")

    # Data and outputs
    p.add_argument("--data_path", type=str, required=True,
                   help="Root directory of your dataset")
    p.add_argument("--out_root", type=str, default="results",
                   help="Root directory for outputs; each stage will create a subdir like res-<mod>/")
    p.add_argument("--stages", type=str, default="t1,t2,flair,t1ce",
                   help="Training order (comma-separated), e.g., t1,t2,flair,t1ce")

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
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--images_rate", type=float, default=1.0)
    p.add_argument("--base_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=3e-4)
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

    p.add_argument("--in_channels", type=int, default=1,
                   help="Input channels (set 4 if you want 4-channel inputs)")

    p.add_argument("--workers_train", type=int, default=8)
    p.add_argument("--workers_val", type=int, default=4)
    p.add_argument("--gpus", type=str, default="0",
                   help="CUDA_VISIBLE_DEVICES value, e.g., 0 or 0,1")
    p.add_argument("--seed", type=int, default=1111)

    return p.parse_args()


def _expand_lists(stages, lists_arg: str, fmt: str):
    if lists_arg.strip():
        parts = [s.strip() for s in lists_arg.split(",") if s.strip()]
        if len(parts) == 1:
            return parts * len(stages)
        assert len(parts) == len(stages), "--train_lists/--val_lists length must match --stages (or pass a single path)"
        return parts
    else:
        if "{mod}" in fmt:
            return [fmt.format(mod=m) for m in stages]
        else:
            return [fmt] * len(stages)


def _resolve_lists(stages, train_lists_arg, val_lists_arg, train_fmt, val_fmt):
    train_lists = _expand_lists(stages, train_lists_arg, train_fmt)
    val_lists   = _expand_lists(stages, val_lists_arg,   val_fmt)
    return train_lists, val_lists


def _verify_paths(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            logging.error(f"[List Missing] {p}")
        raise FileNotFoundError("Some list files do not exist. See errors above.")


def main():
    args = parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"[Device] {device}, GPUs={os.environ.get('CUDA_VISIBLE_DEVICES','-')}")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    assert len(stages) >= 1, "Please provide at least one stage (modality)."

    train_lists, val_lists = _resolve_lists(
        stages, args.train_lists, args.val_lists, args.train_fmt, args.val_fmt
    )
    _verify_paths(train_lists + val_lists)

    os.makedirs(args.out_root, exist_ok=True)

    seen = []
    prev_base_dir = None

    for i, mod in enumerate(stages):
        base_dir = os.path.join(args.out_root, f"res-{mod}")
        prev_img_modes = seen.copy() if seen else None

        cfg = StageConfig(
            base_dir=base_dir,
            data_path=args.data_path,
            train_list=train_lists[i],
            val_list=val_lists[i],
            img_mode=mod,

            prev_img_modes=prev_img_modes,
            prev_base_dir=prev_base_dir,

            mem_size=args.mem_size,
            p_keep=args.p_keep,

            max_epoch=args.max_epoch,
            batch_size=args.batch_size,
            images_rate=args.images_rate,

            base_lr=args.base_lr,
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
            num_workers_train=args.workers_train,
            num_workers_val=args.workers_val,
            device=device,
            seed=args.seed
        )

        logging.info(f"[Stage {i+1}/{len(stages)}] modality={mod}")
        if prev_img_modes:
            logging.info(f"  prev_img_modes: {prev_img_modes}")
        if prev_base_dir:
            logging.info(f"  prev_base_dir: {prev_base_dir}")

        run_stage(cfg)

        # Chain for next stage
        seen.append(mod)
        prev_base_dir = base_dir

    logging.info("\n[All Done] All stages finished successfully.")


if __name__ == "__main__":
    main()
