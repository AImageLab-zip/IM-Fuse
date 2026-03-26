import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
import numpy as np
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from torch import autocast, nn
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
import random
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
import torch.nn.functional as F
from einops import rearrange
masks_total = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
               [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], 
               [True, False, False, True], [True, True, False, False], [True, True, True, False], [True, False, True, True], 
               [True, True, False, True], [False, True, True, True], [True, True, True, True]]

def calculate_mse_loss(original_tokens, reconstructed_tokens, presences):
    mse_losses = []
    for orig_token, recons_token, presence in zip(original_tokens, reconstructed_tokens, presences):
        if not presence:  # 如果该模态在原始输入中是缺失的
            mse_loss = F.mse_loss(recons_token, orig_token)
            mse_losses.append(mse_loss)
    return mse_losses
class nnUNetTrainerMissingRecon(nnUNetTrainer):
    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        selected_mask = random.choice(masks_total)
        
        # for i, mask in enumerate(selected_mask):
        #     if not mask:
        #     for    data[:, i, :, :, :] = 0  # 将选定模态的所有值设置为0
        # assert not torch.all(torch.eq(data, 0)), "All elements in x are zero"
        
        data = data.to(self.device, non_blocking=True)
        
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output, recon, recon_Gt = self.network(data, selected_mask)
            #self.print_to_log_file("selected mask: %s" % str(selected_mask))
            # del data
            l = self.loss(output, target)
            recon = rearrange(recon, 'b (m c) h w d -> b m c h w d', m=4)
            mseloss = calculate_mse_loss(recon_Gt, recon, selected_mask)
            #print('mesloss:', mesloss)
            combined_loss = l + sum(mseloss)
            print('mse_loss:', sum(mseloss))
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            #use tensorboard to log the ce_loss and dc_loss and mse_loss
            #self.writer.add_scalar('Train/CE_loss', ce_loss, self.tr_step)
            #self.writer.add_scalar('Train/DC_loss', dc_loss, self.tr_step)
            #self.writer.add_scalar('Train/MSE_loss', sum(mseloss), self.tr_step)
            #print('ce_loss:', ce_loss.detach().cpu().numpy(), 'dc_loss:', dc_loss.detach().cpu().numpy(), 'mse_loss:', sum(mseloss).detach().cpu().numpy())
        return {'loss': l.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        
        selected_mask = random.choice(masks_total)
        # for i, mask in enumerate(selected_mask):
        #     if not mask:
        #         data[:, i, :, :, :] = 0  # 将选定模态的所有值设置为0
                
        # assert not torch.all(torch.eq(data, 0)), "All elements in x are zero"
        data = data.to(self.device, non_blocking=True)
        
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output,  recon, recon_Gt = self.network(data, selected_mask)
            #print('skip:',skip.shape)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}


