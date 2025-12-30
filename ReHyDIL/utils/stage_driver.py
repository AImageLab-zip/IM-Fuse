# -*- coding: utf-8 -*-
import os
import math
import logging
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import random
import re
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from nets.cph import CPH
from losses import focal_tversky
from dataloader.dataset import BaseDataSets, PatientBatchSampler
from utils.metrics import dice as dice_all
from utils.metrics import batch_dice
from utils.tac_queue import TACWithQueues, build_teacher_queue, BalanceQueue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

@dataclass
class StageConfig:
    base_dir: str
    data_path: str
    train_list: str
    val_list: str
    img_mode: str
    prev_img_modes: Optional[List[str]] = None
    prev_base_dir: Optional[str] = None
    mem_size: int = 320
    p_keep: float = 0.10
    max_epoch: int = 81
    batch_size: int = 16
    images_rate: float = 1.0
    base_lr: float = 1e-3
    weight_decay: float = 3e-4
    optim_name: str = "adam"
    lr_scheduler: str = "warmupMultistep"
    step_num_lr: int = 4
    tversky_w: float = 7.0
    imb_w: float = 8.0
    nce_weight: float = 3.5
    alpha: float = 0.7
    beta: float = 1.5
    gamma: float = 1.2
    in_channels: int = 1
    num_workers_train: int = 8
    num_workers_val: int = 4
    device: str = "cuda"
    seed: int = 1111

class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=1.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth
    def forward(self, inputs, targets):
        p = inputs.reshape(-1)
        q = targets.reshape(-1)
        tp = (p * q).sum()
        fp = ((1 - q) * p).sum()
        fn = (q * (1 - p)).sum()
        return 1.0 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

def _set_seed(seed: int):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def _build_optimizer(net: torch.nn.Module, cfg: StageConfig):
    name = cfg.optim_name.lower()
    if name == "adam":
        return optim.Adam(net.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    if name == "adamw":
        return optim.AdamW(net.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return optim.SGD(net.parameters(), lr=cfg.base_lr, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optim_name}")

def _build_scheduler(optimizer, cfg: StageConfig):
    warm_up_epochs = int(cfg.max_epoch * 0.1)
    if cfg.lr_scheduler == 'warmupMultistep':
        if cfg.step_num_lr == 2:
            ms = [int(cfg.max_epoch * 0.3), int(cfg.max_epoch * 0.6)]
        elif cfg.step_num_lr == 3:
            ms = [int(cfg.max_epoch * 0.25), int(cfg.max_epoch * 0.4), int(cfg.max_epoch * 0.6)]
        else:
            ms = [int(cfg.max_epoch * 0.15), int(cfg.max_epoch * 0.35), int(cfg.max_epoch * 0.55), int(cfg.max_epoch * 0.7)]
        def lr_lambda(epoch):
            if epoch < max(1, warm_up_epochs):
                return (epoch + 1) / max(1, warm_up_epochs)
            steps = sum(int(m <= epoch) for m in ms)
            return 0.1 ** steps
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif cfg.lr_scheduler == 'warmupCosine':
        def lr_lambda(epoch):
            if epoch < max(1, warm_up_epochs):
                return (epoch + 1) / max(1, warm_up_epochs)
            return 0.5 * (math.cos((epoch - warm_up_epochs) / max(1, cfg.max_epoch - warm_up_epochs) * math.pi) + 1)
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif cfg.lr_scheduler == 'autoReduce':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True, cooldown=2, min_lr=0.0)
    else:
        raise ValueError(f"Unknown lr_scheduler: {cfg.lr_scheduler}")

def _load_prev_model(prev_base_dir: Optional[str], device: str) -> Optional[torch.nn.Module]:
    if not prev_base_dir:
        return None
    model_path = os.path.join(prev_base_dir, 'model_CPH_best.pth')
    if not os.path.exists(model_path):
        return None
    model = CPH(n_classes=3).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def _init_from_prev_weights(net: torch.nn.Module, prev_base_dir: Optional[str], device: str):
    if not prev_base_dir:
        return
    model_path = os.path.join(prev_base_dir, 'model_CPH_best.pth')
    if not os.path.exists(model_path):
        return
    state = torch.load(model_path, map_location=device)
    try:
        net.load_state_dict(state, strict=False)
    except:
        pass

def _discover_prev_modalities(current_mode: str) -> List[str]:
    rb_dir = Path('replay_buffer')
    if not rb_dir.exists():
        return []
    modes = []
    pat = re.compile(r"^replay_buffer_(.+)\.pth$")
    for f in rb_dir.iterdir():
        m = pat.match(f.name)
        if m:
            mm = m.group(1)
            if mm != current_mode:
                modes.append(mm)
    return sorted(list(dict.fromkeys(modes)))

def _load_replay(modality: str, device: str = 'cpu') -> Optional[dict]:
    path = os.path.join('replay_buffer', f'replay_buffer_{modality}.pth')
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device)
    ckpt.setdefault('patients', ["unk"] * ckpt['images'].shape[0])
    ckpt.setdefault('modalities', [modality] * ckpt['images'].shape[0])
    return ckpt

def _save_replay(modality: str, images: torch.Tensor, masks: torch.Tensor, losses: torch.Tensor, patients: List[str], modalities: List[str]):
    os.makedirs('replay_buffer', exist_ok=True)
    path = os.path.join('replay_buffer', f'replay_buffer_{modality}.pth')
    torch.save({'images': images.cpu(), 'masks': masks.cpu(), 'losses': losses.cpu(), 'patients': patients, 'modalities': modalities}, path)
    logging.info(f"[Replay] Saved {images.shape[0]} samples to {path}")

def run_stage(cfg: StageConfig):

    os.makedirs(cfg.base_dir, exist_ok=True)
    _set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(cfg.base_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(cfg.base_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    train_ds = BaseDataSets(cfg.data_path, "train", cfg.img_mode, 'masks_all', cfg.train_list, cfg.images_rate)
    val_ds   = BaseDataSets(cfg.data_path, "val",   cfg.img_mode, 'masks_all', cfg.val_list)

    train_sampler = PatientBatchSampler(train_ds.sample_list, cfg.batch_size)

    def _seed_worker(_: int):
        return

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler,
        num_workers=cfg.num_workers_train, pin_memory=True, persistent_workers=True,
        worker_init_fn=_seed_worker
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers_val, pin_memory=True, persistent_workers=True,
        drop_last=False,
        worker_init_fn=_seed_worker
    )

    if cfg.in_channels == 1:
        net = CPH(n_classes=3).to(device)
    else:
        class InputAdapter(torch.nn.Module):
            def __init__(self, k: int):
                super().__init__()
                self.conv = torch.nn.Conv2d(k, 1, kernel_size=1, bias=False)
                with torch.no_grad():
                    self.conv.weight[:] = 1.0 / k
            def forward(self, x):
                return self.conv(x)
        net = torch.nn.Sequential(InputAdapter(cfg.in_channels), CPH(n_classes=3)).to(device)

    _init_from_prev_weights(net, cfg.prev_base_dir, str(device))

    optimizer = _build_optimizer(net, cfg)
    scheduler = _build_scheduler(optimizer, cfg)

    tversky = TverskyLoss(alpha=cfg.alpha, beta=cfg.beta)
    tac_loss = TACWithQueues(alpha=cfg.alpha, beta=cfg.beta, tau=1.0)

    prev_model = _load_prev_model(cfg.prev_base_dir, str(device))

    modes_to_load = cfg.prev_img_modes if (cfg.prev_img_modes and len(cfg.prev_img_modes) > 0) else _discover_prev_modalities(cfg.img_mode)

    teacher_queue: Optional[BalanceQueue] = None
    if prev_model is not None and modes_to_load:
        imgs_list, pats_list, mods_list = [], [], []
        for m in modes_to_load:
            rb = _load_replay(m, device='cpu')
            if rb is None:
                continue
            imgs_list.append(rb['images'].float())
            pats_list.extend(list(rb['patients']))
            mods_list.extend(list(rb['modalities']))
        if imgs_list:
            R_imgs = torch.cat(imgs_list, dim=0)
            teacher_queue = build_teacher_queue(
                prev_model,
                images=R_imgs,
                patient_ids=pats_list,
                modality_ids=mods_list,
                max_size=R_imgs.shape[0],
                batch_size=max(16, cfg.batch_size),
                device=str(device),
                out_device='cpu',
                dtype=torch.float16,
                in_channels=cfg.in_channels
            )
            logging.info(f"[BalanceQueue] QR built with {teacher_queue.size} items; mods={teacher_queue.debug_modality_hist()}")

    omega = 0.0 if (prev_model is None) else 1.0

    current_queue: Optional[BalanceQueue] = None

    stage_names = []
    stage_losses = []

    best_avg3 = 0.0

    best_loss_by_name = {}

    for epoch in range(cfg.max_epoch):
        display_epoch = epoch + 1

        net.train()
        train_loss_sum = 0.0
        train_batches = 0

        train_dice_sum = 0.0
        train_dice_n   = 0

        train_WT_sum = 0.0
        train_TC_sum = 0.0
        train_ET_sum = 0.0
        train_class_batches = 0

        for batch in train_loader:
            imgs_orig = batch['image']
            imgs = imgs_orig.to(device=device, dtype=torch.float32)
            masks = batch['mask'].to(device=device, dtype=torch.float32)
            slice_ids = batch['idx']
            patients  = [str(s).split('_')[0] for s in slice_ids]
            modalities = [cfg.img_mode] * len(patients)

            if cfg.in_channels > 1 and imgs.shape[1] == 1:
                imgs = imgs.repeat(1, cfg.in_channels, 1, 1)

            if imgs.size(0) < cfg.batch_size:
                continue

            probs = torch.sigmoid(net(imgs))

            if current_queue is None:
                _, C, H, W = probs.shape
                current_queue = BalanceQueue(max_size=cfg.mem_size, channels=C, height=H, width=W, dtype=torch.float16, device='cpu')

            loss_tv = tversky(probs, masks)
            loss_ft = focal_tversky(probs, masks, alpha=cfg.alpha, gamma=cfg.gamma, smooth=1.0)

            loss_tac = probs.new_tensor(0.0)
            if omega > 0.0 and teacher_queue is not None and teacher_queue.size > 0:
                loss_tac = tac_loss(probs, patients, modalities, teacher_queue, current_queue)

            loss = cfg.tversky_w * loss_tv + cfg.imb_w * loss_ft + cfg.nce_weight * omega * loss_tac

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            current_queue.enqueue(probs.detach().to('cpu', dtype=torch.float16), patients, modalities)

            pred = (probs > 0.5).float()
            dsum, ncnt = batch_dice(pred.detach().cpu(), masks.detach().cpu())
            train_dice_sum += float(dsum)
            train_dice_n   += int(ncnt)

            p = pred.detach().cpu().numpy().astype('uint8')
            t = masks.detach().cpu().numpy().astype('uint8')
            d0, _, _ = dice_all(p[:, 0], t[:, 0])
            d1, _, _ = dice_all(p[:, 1], t[:, 1])
            d2, _, _ = dice_all(p[:, 2], t[:, 2])
            train_WT_sum += float(d0); train_TC_sum += float(d1); train_ET_sum += float(d2)
            train_class_batches += 1

            train_loss_sum += float(loss.item())
            train_batches  += 1

            with torch.no_grad():
                B = imgs_orig.shape[0]
                for b in range(B):
                    lt = tversky(probs[b:b+1], masks[b:b+1])
                    lf = focal_tversky(probs[b:b+1], masks[b:b+1], alpha=cfg.alpha, gamma=cfg.gamma, smooth=1.0)
                    sup_loss = float((cfg.tversky_w * lt + cfg.imb_w * lf).item())
                    name = str(slice_ids[b])
                    prev = best_loss_by_name.get(name)
                    if (prev is None) or (sup_loss < prev):
                        best_loss_by_name[name] = sup_loss

        train_loss = train_loss_sum / max(1, train_batches)
        train_dice_micro = (train_dice_sum / max(1, train_dice_n)) if train_dice_n > 0 else 0.0
        train_WT_avg = train_WT_sum / max(1, train_class_batches)
        train_TC_avg = train_TC_sum / max(1, train_class_batches)
        train_ET_avg = train_ET_sum / max(1, train_class_batches)
        train_dice_avg = (train_WT_avg + train_TC_avg + train_ET_avg) / 3.0

        net.eval()
        val_loss_sum = 0.0
        val_batches = 0
        val_dice_sum = 0.0
        val_dice_n   = 0
        WT_sum = 0.0; TC_sum = 0.0; ET_sum = 0.0
        class_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                masks = batch['mask'].to(device=device, dtype=torch.float32)
                if cfg.in_channels > 1 and imgs.shape[1] == 1:
                    imgs = imgs.repeat(1, cfg.in_channels, 1, 1)

                probs = torch.sigmoid(net(imgs))
                pred  = (probs > 0.5).float()

                dsum, ncnt = batch_dice(pred.detach().cpu(), masks.detach().cpu())
                val_dice_sum += float(dsum)
                val_dice_n   += int(ncnt)

                loss_val_tv = tversky(probs, masks)
                loss_val_ft = focal_tversky(probs, masks, alpha=cfg.alpha, gamma=cfg.gamma, smooth=1.0)
                val_loss_sum += float((cfg.tversky_w * loss_val_tv + cfg.imb_w * loss_val_ft).item())
                val_batches  += 1

                p = pred.detach().cpu().numpy().astype('uint8')
                t = masks.detach().cpu().numpy().astype('uint8')
                d0, _, _ = dice_all(p[:, 0], t[:, 0])
                d1, _, _ = dice_all(p[:, 1], t[:, 1])
                d2, _, _ = dice_all(p[:, 2], t[:, 2])
                WT_sum += float(d0); TC_sum += float(d1); ET_sum += float(d2)
                class_batches += 1

        val_loss = val_loss_sum / max(1, val_batches)
        val_dice_micro = (val_dice_sum / max(1, val_dice_n)) if val_dice_n > 0 else 0.0
        WT_avg = WT_sum / max(1, class_batches)
        TC_avg = TC_sum / max(1, class_batches)
        ET_avg = ET_sum / max(1, class_batches)
        val_dice_avg = (WT_avg + TC_avg + ET_avg) / 3.0
        avg3 = val_dice_avg

        if cfg.lr_scheduler == 'autoReduce':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        logging.info(
            f"[{cfg.img_mode}] Epoch {display_epoch:03d}/{cfg.max_epoch:03d} | "
            f"train_loss={train_loss:.4f} "
            f"train_dice_micro={train_dice_micro:.4f} train_dice_avg={train_dice_avg:.4f} "
            f"(WT={train_WT_avg:.4f} TC={train_TC_avg:.4f} ET={train_ET_avg:.4f}) | "
            f"val_loss={val_loss:.4f} "
            f"val_dice_micro={val_dice_micro:.4f} val_dice_avg={val_dice_avg:.4f} "
            f"(WT={WT_avg:.4f} TC={TC_avg:.4f} ET={ET_avg:.4f}) | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if avg3 > best_avg3:
            best_avg3 = avg3
            torch.save(net.state_dict(), os.path.join(cfg.base_dir, 'model_CPH_best.pth'))
            logging.info(f"[Best] epoch={display_epoch} val_dice_avg={best_avg3:.4f} val_dice_micro={val_dice_micro:.4f}")

    if len(best_loss_by_name) > 0:

        names = list(best_loss_by_name.keys())
        losses = np.array([best_loss_by_name[n] for n in names], dtype=np.float32)

        # Keep the same selection policy as before (median-deviation top-k), now on deduplicated slices
        k = max(1, int(cfg.p_keep * len(losses)))
        mu = np.median(losses)
        dev = np.abs(losses - mu)
        sel_idx = np.argsort(dev)[:k]
        sel_names = [names[i] for i in sel_idx]
        sel_losses = losses[sel_idx]


        img_dir = Path(cfg.data_path) / f"imgs_{cfg.img_mode}"
        mask_dir = Path(cfg.data_path) / "masks_all"

        def _to_chw(a: np.ndarray):
            if a.ndim == 2:
                return a[None, ...]
            return a

        # Chunked I/O to be memory friendly
        X_chunks, Y_chunks = [], []
        CHUNK = 1024
        for s in range(0, len(sel_names), CHUNK):
            part = sel_names[s:s+CHUNK]
            imgs_np = [_to_chw(np.load(img_dir / f"{n}.npy")) for n in part]
            masks_np = [_to_chw(np.load(mask_dir / f"{n}.npy")) for n in part]
            X_chunks.append(torch.from_numpy(np.stack(imgs_np)).to(torch.float16))
            Y_chunks.append(torch.from_numpy(np.stack(masks_np)).to(torch.uint8))

        Xs = torch.cat(X_chunks, dim=0).contiguous()
        Ys = torch.cat(Y_chunks, dim=0).contiguous()
        Ls = torch.from_numpy(sel_losses).to(torch.float32)
        pats = [str(n).split('_')[0] for n in sel_names]
        mods = [cfg.img_mode] * len(sel_names)

        _save_replay(cfg.img_mode, Xs, Ys, Ls, pats, mods)
        logging.info(f"[Replay] unique={len(names)} keep%={cfg.p_keep:.2f} -> saved={len(sel_names)}")

        stage_names.append(cfg.img_mode)
        stage_losses.append(float(best_avg3))
        best_loss_by_name.clear()
        best_loss_by_name.clear()


    torch.save(net.state_dict(), os.path.join(cfg.base_dir, 'model_CPH_last.pth'))
    logging.info("[Stage] Done.")
    logger.removeHandler(fh)
    fh.close()
