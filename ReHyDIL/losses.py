import torch
import torch.nn.functional as F

epsilon = 1e-5
smooth = 1.0

def dsc(y_pred, y_true, smooth=1.0):

    y_pred_f = y_pred.view(y_pred.size(0), -1)
    y_true_f = y_true.view(y_true.size(0), -1)
    intersection = (y_pred_f * y_true_f).sum(dim=1)
    score = (2. * intersection + smooth) / (y_pred_f.sum(dim=1) + y_true_f.sum(dim=1) + smooth)
    return score.mean()

def dice_loss(y_pred, y_true, smooth=1.0):

    return 1 - dsc(y_pred, y_true, smooth)

def bce_dice_loss(y_pred, y_true, smooth=1.0):

    bce = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
    d_loss = dice_loss(y_pred, y_true, smooth)
    return bce + d_loss

def confusion_metrics(y_pred, y_true, smooth=1.0):

    y_pred = torch.clamp(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred
    y_true_neg = 1 - y_true

    tp = (y_pred * y_true).sum(dim=[1,2,3])
    fp = (y_pred * y_true_neg).sum(dim=[1,2,3])
    fn = (y_pred_neg * y_true).sum(dim=[1,2,3])

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return precision.mean(), recall.mean()

def true_positive(y_pred, y_true, smooth=1.0):

    y_pred_pos = torch.round(torch.clamp(y_pred, 0, 1))
    y_true_pos = torch.round(torch.clamp(y_true, 0, 1))
    tp = (y_pred_pos * y_true_pos).sum(dim=[1,2,3])
    return ((tp + smooth) / (y_true_pos.sum(dim=[1,2,3]) + smooth)).mean()

def true_negative(y_pred, y_true, smooth=1.0):

    y_pred_pos = torch.round(torch.clamp(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_true_neg = 1 - torch.round(torch.clamp(y_true, 0, 1))
    tn = (y_pred_neg * y_true_neg).sum(dim=[1,2,3])
    return ((tn + smooth) / (y_true_neg.sum(dim=[1,2,3]) + smooth)).mean()


def tversky(y_pred, y_true, alpha=0.7, smooth=1.0):

    if y_pred.size(0) != y_true.size(0):
        return torch.tensor(1e-8, device=y_pred.device, dtype=y_pred.dtype)

    y_pred_f = y_pred.reshape(y_pred.size(0), -1)
    y_true_f = y_true.reshape(y_true.size(0), -1)
    true_pos = (y_pred_f * y_true_f).sum(dim=1)
    false_neg = (y_true_f * (1 - y_pred_f)).sum(dim=1)
    false_pos = (y_pred_f * (1 - y_true_f)).sum(dim=1)
    tversky_index = (true_pos + smooth) / (
        true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth
    )
    return tversky_index.mean()

def tversky_loss(y_pred, y_true, alpha=0.7, smooth=1.0):

    return 1 - tversky(y_pred, y_true, alpha, smooth)

def focal_tversky(y_pred, y_true, alpha=0.7, gamma=1.5, smooth=1.0):

    tversky_index = tversky(y_pred, y_true, alpha, smooth)
    return torch.pow((1 - tversky_index), gamma)

