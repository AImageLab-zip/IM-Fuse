import torch
import torch.nn as nn
import torchvision.models as models

# copy from https://github.com/zzbdr/DL/blob/main/Super-resolution/SRGAN/loss.py

class VGG(nn.Module):
    def __init__(self, device):
        super(VGG, self).__init__()
        vgg = models.vgg19(True)
        for pa in vgg.parameters():
            pa.requires_grad = False
        self.vgg = vgg.features[:16]
        self.vgg = self.vgg.to(device)

    def forward(self, x):
        out = self.vgg(x)
        return out


class ContentLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mse = nn.MSELoss()
        self.vgg19 = VGG(device)

    def forward(self, fake, real):
        feature_fake = self.vgg19(fake)
        feature_real = self.vgg19(real)
        loss = self.mse(feature_fake, feature_real)
        return loss

'''
class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        loss = torch.sum(-torch.log(x))
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg_loss = ContentLoss(device)
        self.adversarial = AdversarialLoss()

    def forward(self, fake, real, x):
        vgg_loss = self.vgg_loss(fake, real)
        adversarial_loss = self.adversarial(x)
        return vgg_loss + 1e-3*adversarial_loss


class RegularizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        a = torch.square(
            x[:, :, :x.shape[2]-1, :x.shape[3]-1] - x[:, :, 1:x.shape[2], :x.shape[3]-1]
        )
        b = torch.square(
            x[:, :, :x.shape[2]-1, :x.shape[3]-1] - x[:, :, :x.shape[2]-1, 1:x.shape[3]]
        )
        loss = torch.sum(torch.pow(a+b, 1.25))
        return loss
'''

if __name__ == '__main__':
    import numpy as np
    c = ContentLoss("cpu")
    f1 = torch.rand([1, 3, 64, 64])
    f1 = torch.from_numpy(np.ones([1,3,64,64,64])).float()
    r1 = torch.rand([1, 3, 64, 64])
    r1 = torch.from_numpy(np.ones([1,3,64,64,64])).float()
    print(c(f1, r1))
    # # ad = AdversarialLoss()
    # i = torch.rand([1, 2, 1, 1])
    # # print(ad(i))
    # p = PerceptualLoss()
    # print(p(f1, r1, i))
    # img = torch.rand([1, 3, 64, 64])
    # r = RegularizationLoss()
    # print(r(img))









