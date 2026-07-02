from __future__ import annotations

import torch
import torch.nn.functional as F

from mimose.losses.imfuse import dice_loss, softmax_weighted_loss


class CLRSLoss:
    """Loss wrapper matching CLRS's multi-term training objective: fused
    segmentation Dice + CE, a contrastive loss over per-modality projection
    vectors, and a cross-entropy loss classifying which modality each
    projection vector came from."""

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        num_cls: int | None = None,
        coe_specloss: float = 0.1,
        coe_consist: float = 0.1,
        temperature: float = 0.05,
        eps: float = 1e-7,
        log_clamp_min: float = 0.005,
    ) -> None:
        resolved_num_classes = (
            int(num_classes)
            if num_classes is not None
            else int(num_cls) if num_cls is not None else 4
        )
        self.num_classes = resolved_num_classes
        self.coe_specloss = float(coe_specloss)
        self.coe_consist = float(coe_consist)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.log_clamp_min = float(log_clamp_min)

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.softmax_weighted_loss(output, target) + self.dice_loss(output, target)

    def training_loss(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]],
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fused_logits, spec_logits_cat, spec_info_vector, _generated_map = outputs

        fuse_pred = F.softmax(fused_logits, dim=1)
        fuse_cross = self.softmax_weighted_loss(fuse_pred, target)
        fuse_dice = self.dice_loss(fuse_pred, target)
        fuse_loss = fuse_cross + fuse_dice

        spec_modal_loss = self._contrastive_modal_loss(spec_info_vector)
        spec_logit_cls_loss = self._modality_classification_loss(spec_logits_cat, mask)

        total = fuse_loss + (self.coe_specloss * spec_modal_loss) + (self.coe_consist * spec_logit_cls_loss)

        return {
            "loss": total,
            "fusecross": fuse_cross,
            "fusedice": fuse_dice,
            "specmodalloss": spec_modal_loss,
            "speclogitclsloss": spec_logit_cls_loss,
        }

    def dice_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(output, target, num_cls=self.num_classes, eps=self.eps)

    def softmax_weighted_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return softmax_weighted_loss(
            output,
            target,
            num_cls=self.num_classes,
            log_clamp_min=self.log_clamp_min,
            eps=self.eps,
        )

    def _contrastive_modal_loss(self, spec_info_vector: torch.Tensor) -> torch.Tensor:
        """Cosine-similarity contrastive loss over per-modality projection
        vectors (ports ``loss_contrastive_modal_fn`` from
        ``legacy/CLRS/segmentor/model_trainer.py``, batched per-sample
        instead of assuming ``batch_size == 1``)."""
        vectors = spec_info_vector.permute(1, 0, 2)
        anchor = F.normalize(vectors, p=2, dim=-1)
        noisy = F.normalize(vectors + torch.randn_like(vectors), p=2, dim=-1)

        logits = torch.matmul(anchor, noisy.transpose(-2, -1)) / self.temperature
        num_modality = logits.size(-1)
        eye = torch.eye(num_modality, device=logits.device, dtype=logits.dtype)

        p_iv = (eye * logits).sum(dim=-1)
        sum_p_iv = torch.exp(logits).sum(dim=-1)
        log_prob = p_iv - torch.log(sum_p_iv + 1e-6)
        return -log_prob.mean()

    @staticmethod
    def _modality_classification_loss(
        spec_logits_cat: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        num_modality, batch_size, _ = spec_logits_cat.shape
        device = spec_logits_cat.device

        total = spec_logits_cat.new_tensor(0.0)
        num_supervised = 0
        for slot in range(num_modality):
            present = (
                mask[:, slot] if mask is not None else torch.ones(batch_size, dtype=torch.bool, device=device)
            )
            if not bool(present.any()):
                continue
            logits = spec_logits_cat[slot][present]
            labels = torch.full((logits.size(0),), slot, dtype=torch.long, device=device)
            total = total + F.cross_entropy(logits, labels)
            num_supervised += 1

        return total / max(num_supervised, 1)


__all__ = ["CLRSLoss"]
