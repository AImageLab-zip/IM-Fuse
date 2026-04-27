import torch.nn.functional as F
import torch
import logging
import torch.nn as nn


__all__ = [
    "sigmoid_dice_loss",
    "softmax_dice_loss",
    "GeneralizedDiceLoss",
    "FocalLoss",
    "dice_loss",
]

cross_entropy = F.cross_entropy


def mse_loss(output, target):
    MSE = nn.MSELoss()
    return MSE(output, target)


def cross_entropy_loss(output, target):
    CEL = nn.CrossEntropyLoss()
    return CEL(output, target)


def l2_regularization_loss(w):
    return torch.square(w).sum()


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="sum") * (self.T**2) / y_s.shape[0]
        return loss


def MMD_loss(x, y, width=1):
    x_n = x.shape[0]
    y_n = y.shape[0]

    x = x.reshape(x.size(0), x.size(1) * x.size(2) * x.size(3) * x.size(4)).contiguous()
    y = y.reshape(y.size(0), y.size(1) * y.size(2) * y.size(3) * y.size(4)).contiguous()

    x_square = torch.sum(x * x, 1)
    y_square = torch.sum(y * y, 1)

    kxy = torch.matmul(x, y.t())
    kxy = kxy - 0.5 * x_square.unsqueeze(1).expand(x_n, y_n)
    kxy = kxy - 0.5 * y_square.expand(x_n, y_n)
    kxy = torch.exp(width * kxy).sum() / x_n / y_n

    kxx = torch.matmul(x, x.t())
    kxx = kxx - 0.5 * x_square.expand(x_n, x_n)
    kxx = kxx - 0.5 * x_square.expand(x_n, x_n)
    kxx = torch.exp(width * kxx).sum() / x_n / x_n

    kyy = torch.matmul(y, y.t())
    kyy = kyy - 0.5 * y_square.expand(y_n, y_n)
    kyy = kyy - 0.5 * y_square.expand(y_n, y_n)
    kyy = torch.exp(width * kyy).sum() / y_n / y_n

    return kxx + kyy - 2 * kxy


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size=2, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).cuda())
        self.register_buffer(
            "negatives_mask",
            (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).cuda()).float(),
        )

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(
            similarity_matrix / self.temperature
        )  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


def Dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


def dice_loss(output, target, num_cls=5, eps=1e-7):
    target = target.float()
    for i in range(num_cls):
        num = torch.sum(output[:, i, :, :, :] * target[:, i, :, :, :])
        l = torch.sum(output[:, i, :, :, :])
        r = torch.sum(target[:, i, :, :, :])
        if i == 0:
            dice = 2.0 * num / (l + r + eps)
        else:
            dice += 2.0 * num / (l + r + eps)
    return 1.0 - 1.0 * dice / num_cls


def softmax_weighted_loss(output, target, num_cls=5):
    target = target.float()
    B, _, H, W, Z = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        weighted = 1.0 - (
            torch.sum(targeti, (1, 2, 3)) * 1.0 / torch.sum(target, (1, 2, 3, 4))
        )
        weighted = torch.reshape(weighted, (-1, 1, 1, 1)).repeat(1, H, W, Z)
        if i == 0:
            cross_loss = (
                -1.0
                * weighted
                * targeti
                * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
            )
        else:
            cross_loss += (
                -1.0
                * weighted
                * targeti
                * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
            )
    cross_loss = torch.mean(cross_loss)
    return cross_loss


def softmax_loss(output, target, num_cls=5):
    target = target.float()
    _, _, H, W, Z = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        if i == 0:
            cross_loss = (
                -1.0
                * targeti
                * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
            )
        else:
            cross_loss += (
                -1.0
                * targeti
                * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
            )
    cross_loss = torch.mean(cross_loss)
    return cross_loss


def FocalLoss(output, target, alpha=0.25, gamma=2.0):
    target[target == 4] = 3  # label [4] -> [3]
    # target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4,H,W,D]
    if output.dim() > 2:
        output = output.view(
            output.size(0), output.size(1), -1
        )  # N,C,H,W,D => N,C,H*W*D
        output = output.transpose(1, 2)  # N,C,H*W*D => N,H*W*D,C
        output = output.contiguous().view(-1, output.size(2))  # N,H*W*D,C => N*H*W*D,C
    if target.dim() == 5:
        target = target.contiguous().view(target.size(0), target.size(1), -1)
        target = target.transpose(1, 2)
        target = target.contiguous().view(-1, target.size(2))
    if target.dim() == 4:
        target = target.view(-1)  # N*H*W*D
    # compute the negative likelyhood
    logpt = -F.cross_entropy(output, target)
    pt = torch.exp(logpt)
    # compute the loss
    loss = -((1 - pt) ** gamma) * logpt
    # return loss.sum()
    return loss.mean()


def dice(output, target, eps=1e-5):  # soft dice loss
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num / den


def sigmoid_dice_loss(output, target, alpha=1e-5):
    # output: [-1,3,H,W,T]
    # target: [-1,H,W,T] noted that it includes 0,1,2,4 here
    loss1 = dice(output[:, 0, ...], (target == 1).float(), eps=alpha)
    loss2 = dice(output[:, 1, ...], (target == 2).float(), eps=alpha)
    loss3 = dice(output[:, 2, ...], (target == 4).float(), eps=alpha)
    logging.info(
        "1:{:.4f} | 2:{:.4f} | 4:{:.4f}".format(
            1 - loss1.data, 1 - loss2.data, 1 - loss3.data
        )
    )
    return loss1 + loss2 + loss3


def softmax_dice_loss(output, target):
    # output : [bsize,c,H,W,D]
    # target : [bsize,H,W,D]
    loss1 = dice(output[:, 1, ...], (target == 1).float())
    loss2 = dice(output[:, 2, ...], (target == 2).float())
    loss3 = dice(output[:, 3, ...], (target == 4).float())
    logging.info(
        "1:{:.4f} | 2:{:.4f} | 4:{:.4f}".format(
            1 - loss1.data, 1 - loss2.data, 1 - loss3.data
        )
    )

    return loss1 + loss2 + loss3


def GeneralizedDiceLoss(
    output, target, eps=1e-5, weight_type="square"
):  # Generalized dice loss
    """
    Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
    """

    if target.dim() == 4:
        target[target == 4] = 3  # label [4] -> [3]
        target = expand_target(
            target, n_class=output.size()[1]
        )  # [N,H,W,D] -> [N,4，H,W,D]

    output = flatten(output)[
        1:, ...
    ]  # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:, ...]  # [class, N*H*W*D]

    target_sum = target.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == "square":
        class_weights = 1.0 / (target_sum * target_sum + eps)
    elif weight_type == "identity":
        class_weights = 1.0 / (target_sum + eps)
    elif weight_type == "sqrt":
        class_weights = 1.0 / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError("Check out the weight_type :", weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2 * intersect[0] / (denominator[0] + eps)
    loss2 = 2 * intersect[1] / (denominator[1] + eps)
    loss3 = 2 * intersect[2] / (denominator[2] + eps)

    return 1 - 2.0 * intersect_sum / denominator_sum, [
        loss1.data,
        loss2.data,
        loss3.data,
    ]


def expand_target(x, n_class, mode="softmax"):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :return: 5D output image (NxCxDxHxW)
    """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == "softmax":
        xx[:, 1, :, :, :] = x == 1
        xx[:, 2, :, :, :] = x == 2
        xx[:, 3, :, :, :] = x == 3
    if mode.lower() == "sigmoid":
        xx[:, 0, :, :, :] = x == 1
        xx[:, 1, :, :, :] = x == 2
        xx[:, 2, :, :, :] = x == 3
    return xx.to(x.device)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)
