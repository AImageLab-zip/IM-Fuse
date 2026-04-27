from torch import nn

def get_style_discriminator(num_classes, ndf=64):
    return nn.Sequential(
        nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf * 4, 1, kernel_size=(2,3,2), stride=1, padding=0)
    )