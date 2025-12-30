import torch
import torch.nn as nn


class TACLoss(nn.Module):
    def __init__(self, temperature=0.07, ignore_label=-1, alpha=0.7, smooth=1.0, gamma=1.5):
        super(TACLoss, self).__init__()
        self.temperature = temperature
        self.ignore_label = ignore_label
        self.alpha = alpha
        self.smooth = smooth
        self.gamma = gamma

    def global_pool(self, features):
        return torch.mean(features, dim=(2, 3))

    def tversky_similarity(self, f1, f2):
        z1 = f1
        z2 = f2
        z1_exp = z1.unsqueeze(1)
        z2_exp = z2.unsqueeze(0)
        TP = torch.sum(z1_exp * z2_exp, dim=2)
        FP = torch.sum(z1_exp * (1 - z2_exp), dim=2)
        FN = torch.sum((1 - z1_exp) * z2_exp, dim=2)
        sim = (TP + self.smooth) / (TP + self.alpha * FP + (1 - self.alpha) * FN + self.smooth)
        return sim

    def forward(self, features1, features2, labels):
        device = features1.device
        if features1.dim() == 4:
            pooled1 = self.global_pool(features1)
            pooled2 = self.global_pool(features2)
        else:
            pooled1 = features1
            pooled2 = features2

        if labels.dim() == 4 and labels.shape[1] > 1:
            labels = torch.argmax(labels, dim=1)
        elif labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        labels = labels.view(labels.size(0), -1)
        sample_labels = labels[:, 0]

        sim_matrix = self.tversky_similarity(pooled1, pooled2) / self.temperature
        positive_mask = (sample_labels.unsqueeze(1) == sample_labels.unsqueeze(0)).float()

        exp_sim = torch.exp(sim_matrix)
        sum_exp = torch.sum(exp_sim, dim=1, keepdim=True) + 1e-8
        log_prob = sim_matrix - torch.log(sum_exp)

        positive_count = torch.sum(positive_mask, dim=1)
        valid_mask = positive_count > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = torch.sum(positive_mask * log_prob, dim=1) / (positive_count + 1e-8)
        loss = - torch.mean(mean_log_prob_pos[valid_mask])
        return loss
