from torch import nn
from torch import cat
import torch
import torch.nn.functional as F
import time
import torch.nn.init as init

###MFI block
class fianl_diff_code_block(nn.Module):
    def __init__(self,in_channel):
        super(fianl_diff_code_block,self).__init__()
        self.Relation1 = nn.Sequential(
            nn.Linear(in_channel * 4, in_channel*2),
            nn.LeakyReLU(),
            nn.Linear(in_channel*2, in_channel)
            # nn.LeakyReLU(),
        )
        self.Relation2 = nn.Sequential(
            nn.Linear(in_channel * 4, in_channel*2),
            nn.LeakyReLU(),
            nn.Linear(in_channel*2, in_channel)
            # nn.LeakyReLU(),
        )
        self.Relation3 = nn.Sequential(
            nn.Linear(in_channel * 4, in_channel*2),
            nn.LeakyReLU(),
            nn.Linear(in_channel*2, in_channel)
            # nn.LeakyReLU(),
        )
        self.Relation4 = nn.Sequential(
            nn.Linear(in_channel * 4, in_channel*2),
            nn.LeakyReLU(),
            nn.Linear(in_channel*2, in_channel)
            # nn.LeakyReLU(),
        )
    def forward(self, x1,x2,x3,x4,mod_code):     ####mod_code: b*4
        b,c,h,w,l = x1.shape
        X_ori = torch.cat((x1.unsqueeze(1),x2.unsqueeze(1),x3.unsqueeze(1),x4.unsqueeze(1)),1)  ###(b*4*c*h*w*l)
        X = torch.mean(X_ori.view(b,4,c,h*w*l),-1) ##b*4*c
        X1 = X.unsqueeze(1).repeat(1,4,1,1)  ##b*4*4*c
        X2 = X.unsqueeze(2).repeat(1,1,4,1)
        X_R = torch.cat((X1, X2),-1)   ###b*4*4*2c

        mod_code = mod_code.unsqueeze(-1).repeat(1, 1, 2 * c)
        X_R_1, X_R_2, X_R_3, X_R_4 = self.Relation1(torch.cat((X_R[:,0,:,:],mod_code),dim=-1)),self.Relation1(torch.cat((X_R[:,1,:,:],mod_code),dim=-1)),\
                                     self.Relation1(torch.cat((X_R[:,2,:,:],mod_code),dim=-1)),self.Relation1(torch.cat((X_R[:,3,:,:],mod_code),dim=-1)),


        X_R_1,X_R_2,X_R_3,X_R_4 = F.softmax(X_R_1, 1),F.softmax(X_R_2, 1),F.softmax(X_R_3, 1),F.softmax(X_R_4, 1)
        X_1_out = torch.matmul(X_ori.view(b,4,c,h*w*l).permute(0,2,3,1),X_R_1.permute(0,2,1).unsqueeze(-1)).squeeze(-1)  ##b*c*(h*w*l)*4 and b*c*4*1 -> b*c*(h*w*l)*1
        X_2_out = torch.matmul(X_ori.view(b, 4, c, h * w * l).permute(0, 2, 3, 1),
                               X_R_2.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)
        X_3_out = torch.matmul(X_ori.view(b, 4, c, h * w * l).permute(0, 2, 3, 1),
                               X_R_3.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)
        X_4_out = torch.matmul(X_ori.view(b, 4, c, h * w * l).permute(0, 2, 3, 1),
                               X_R_4.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)

        X_1_out,X_2_out,X_3_out,X_4_out = X_1_out.reshape(b,c,h,w,l),X_2_out.reshape(b,c,h,w,l),X_3_out.reshape(b,c,h,w,l),X_4_out.reshape(b,c,h,w,l)

        return x1+X_1_out,x2+X_2_out,x3+X_3_out,x4+X_4_out


###Unet Encoder block
class EnBlock(nn.Module):
    def __init__(self,in_c):
        super(EnBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(in_c//8,in_c),
            nn.ReLU(True),
            nn.Conv3d(in_c, in_c,3,padding=1),
            nn.GroupNorm(in_c//8,in_c),
            nn.ReLU(True),
            nn.Conv3d(in_c, in_c,3,padding=1),
        )
    def forward(self, x):
        res = x
        out = self.conv(x)
        return (out+res)

class EnDown(nn.Module):
    def __init__(self,in_c,ou_c):
        super(EnDown,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c,ou_c,kernel_size=3,stride=2,padding=1)
        )
    def forward(self, x):
        out = self.conv(x)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.init_conv = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3,
                                   padding=1)  # 64,128,128,128
        self.en1 = EnBlock(32)
        self.ed1 = EnDown(32, 64)
        self.en2 = EnBlock(64)
        self.ed2 = EnDown(64, 128)
        self.en3 = EnBlock(128)
        self.ed3 = EnDown(128, 256)
        self.en4 = EnBlock(256)
    def forward(self, x):
        x = self.init_conv(x)
        c1 = self.en1(x)
        p1 = self.ed1(c1)
        c2 = self.en2(p1)
        p2 = self.ed2(c2)
        c3 = self.en3(p2)
        p3 = self.ed3(c3)
        c4 = self.en4(p3)
        return c1,c2,c3,c4

###Unet Dncoder block

class DnUp(nn.Module):
    def __init__(self, in_c, ou_c):
        super(DnUp, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c,in_c,1),
            nn.ConvTranspose3d(in_c,ou_c,2,2)
        )

    def forward(self, x,en_x):
        out = self.conv(x)


        return (out+en_x)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv3d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class no_share_unet(nn.Module):
    def __init__(self,in_channel,out_channel,diff=False,deepSupvision=False):
        super(no_share_unet, self).__init__()
        class_nums = out_channel
        self.is_diff = diff
        self.deepSupvision = deepSupvision


        self.diff1 = fianl_diff_code_block(64)
        self.diff2 = fianl_diff_code_block(128)
        self.diff3 = fianl_diff_code_block(256)
        self.diff4 = fianl_diff_code_block(128)
        self.diff5 = fianl_diff_code_block(64)


        self.init_conv_1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3,
                                   padding=1)  # 64,128,128,128
        self.en1_1 = EnBlock(32)
        self.ed1_1 = EnDown(32, 64)
        self.en2_1 = EnBlock(64)
        self.ed2_1 = EnDown(64, 128)
        self.en3_1 = EnBlock(128)
        self.ed3_1 = EnDown(128, 256)
        self.en4_1 = EnBlock(256)

        self.init_conv_2 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3,
                                     padding=1)  # 64,128,128,128
        self.en1_2 = EnBlock(32)
        self.ed1_2 = EnDown(32, 64)
        self.en2_2 = EnBlock(64)
        self.ed2_2 = EnDown(64, 128)
        self.en3_2 = EnBlock(128)
        self.ed3_2 = EnDown(128, 256)
        self.en4_2 = EnBlock(256)

        self.init_conv_3 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3,
                                     padding=1)  # 64,128,128,128
        self.en1_3 = EnBlock(32)
        self.ed1_3 = EnDown(32, 64)
        self.en2_3 = EnBlock(64)
        self.ed2_3 = EnDown(64, 128)
        self.en3_3 = EnBlock(128)
        self.ed3_3 = EnDown(128, 256)
        self.en4_3 = EnBlock(256)

        self.init_conv_4 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3,
                                     padding=1)  # 64,128,128,128
        self.en1_4 = EnBlock(32)
        self.ed1_4 = EnDown(32, 64)
        self.en2_4 = EnBlock(64)
        self.ed2_4 = EnDown(64, 128)
        self.en3_4 = EnBlock(128)
        self.ed3_4 = EnDown(128, 256)
        self.en4_4 = EnBlock(256)


        self.ud1_1 = DnUp(256, 128)  #
        self.un1_1 = EnBlock(128)
        self.ud2_1 = DnUp(128, 64)
        self.un2_1 = EnBlock(64)
        self.ud3_1 = DnUp(64, 32)
        self.un3_1 = EnBlock(32)

        self.ud1_2 = DnUp(256, 128)  #
        self.un1_2 = EnBlock(128)
        self.ud2_2 = DnUp(128, 64)
        self.un2_2 = EnBlock(64)
        self.ud3_2 = DnUp(64, 32)
        self.un3_2 = EnBlock(32)

        self.ud1_3 = DnUp(256, 128)  #
        self.un1_3 = EnBlock(128)
        self.ud2_3 = DnUp(128, 64)
        self.un2_3 = EnBlock(64)
        self.ud3_3 = DnUp(64, 32)
        self.un3_3 = EnBlock(32)

        self.ud1_4 = DnUp(256, 128)  #
        self.un1_4 = EnBlock(128)
        self.ud2_4 = DnUp(128, 64)
        self.un2_4 = EnBlock(64)
        self.ud3_4 = DnUp(64, 32)
        self.un3_4 = EnBlock(32)

        self.out_conv1 = nn.Conv3d(32,class_nums,1)
        self.out_conv2 = nn.Conv3d(32, class_nums, 1)
        self.out_conv3 = nn.Conv3d(32, class_nums, 1)
        self.out_conv4 = nn.Conv3d(32, class_nums, 1)

        self.cat_out_conv = nn.Sequential(
            nn.Conv3d(4*class_nums,class_nums,1)
        )

        if self.deepSupvision:
            self.stage1Out = nn.Sequential(
                nn.Conv3d(128,class_nums,kernel_size=3,padding=1),
                nn.Upsample(scale_factor=4)

            )
            self.stage2Out = nn.Sequential(
                nn.Conv3d(64,class_nums,kernel_size=3,padding=1),
                nn.Upsample(scale_factor=2),
            )


    def weight_init(self):
        initializer = kaiming_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, x1,x2,x3,x4,mask):
        x1 = self.init_conv_1(x1)
        x2 = self.init_conv_2(x2)
        x3 = self.init_conv_3(x3)
        x4 = self.init_conv_4(x4)


        c1_1 = self.en1_1(x1)
        c1_2 = self.en1_2(x2)
        c1_3 = self.en1_3(x3)
        c1_4 = self.en1_4(x4)



        p1_1 = self.ed1_1(c1_1)
        p1_2 = self.ed1_2(c1_2)
        p1_3 = self.ed1_3(c1_3)
        p1_4 = self.ed1_4(c1_4)
        if self.is_diff:
            p1_1, p1_2, p1_3, p1_4 = self.diff1(p1_1, p1_2, p1_3, p1_4,mask)


        c2_1 = self.en2_1(p1_1)
        c2_2 = self.en2_2(p1_2)
        c2_3 = self.en2_3(p1_3)
        c2_4 = self.en2_4(p1_4)



        p2_1 = self.ed2_1(c2_1)
        p2_2 = self.ed2_2(c2_2)
        p2_3 = self.ed2_3(c2_3)
        p2_4 = self.ed2_4(c2_4)
        if self.is_diff:
            p2_1, p2_2, p2_3, p2_4 = self.diff2(p2_1, p2_2, p2_3, p2_4,mask)


        c3_1 = self.en3_1(p2_1)
        c3_2 = self.en3_2(p2_2)
        c3_3 = self.en3_3(p2_3)
        c3_4 = self.en3_4(p2_4)


        p3_1 = self.ed3_1(c3_1)
        p3_2 = self.ed3_2(c3_2)
        p3_3 = self.ed3_3(c3_3)
        p3_4 = self.ed3_4(c3_4)
        if self.is_diff:
            p3_1, p3_2, p3_3, p3_4 = self.diff3(p3_1, p3_2, p3_3, p3_4,mask)


        c4_1 = self.en4_1(p3_1)
        c4_2 = self.en4_2(p3_2)
        c4_3 = self.en4_3(p3_3)
        c4_4 = self.en4_4(p3_4)

        up5_1 = self.ud1_1(c4_1, c3_1)
        up5_2 = self.ud1_2(c4_2, c3_2)
        up5_3 = self.ud1_3(c4_3, c3_3)
        up5_4 = self.ud1_4(c4_4, c3_4)
        if self.deepSupvision:
            stage1Out1 = self.stage1Out(up5_1)
            stage1Out2 = self.stage1Out(up5_2)
            stage1Out3 = self.stage1Out(up5_3)
            stage1Out4 = self.stage1Out(up5_4)

        un5_1 = self.un1_1(up5_1)
        un5_2 = self.un1_2(up5_2)
        un5_3 = self.un1_3(up5_3)
        un5_4 = self.un1_4(up5_4)

        if self.is_diff:
            un5_1, un5_2, un5_3, un5_4 = self.diff4(un5_1, un5_2, un5_3, un5_4,mask)


        up6_1 = self.ud2_1(un5_1, c2_1)
        up6_2 = self.ud2_2(un5_2, c2_2)
        up6_3 = self.ud2_3(un5_3, c2_3)
        up6_4 = self.ud2_4(un5_4, c2_4)
        if self.deepSupvision:
            stage2Out1 = self.stage2Out(up6_1)
            stage2Out2 = self.stage2Out(up6_2)
            stage2Out3 = self.stage2Out(up6_3)
            stage2Out4 = self.stage2Out(up6_4)

        un6_1 = self.un2_1(up6_1)
        un6_2 = self.un2_2(up6_2)
        un6_3 = self.un2_3(up6_3)
        un6_4 = self.un2_4(up6_4)
        if self.is_diff:
            un6_1, un6_2, un6_3, un6_4 = self.diff5(un6_1, un6_2, un6_3, un6_4,mask)


        up7_1 = self.ud3_1(un6_1, c1_1)
        up7_2 = self.ud3_2(un6_2, c1_2)
        up7_3 = self.ud3_3(un6_3, c1_3)
        up7_4 = self.ud3_4(un6_4, c1_4)

        un7_1 = self.un3_1(up7_1)
        un7_2 = self.un3_2(up7_2)
        un7_3 = self.un3_3(up7_3)
        un7_4 = self.un3_4(up7_4)

        out1 = self.out_conv1(un7_1)
        out2 = self.out_conv2(un7_2)
        out3 = self.out_conv3(un7_3)
        out4 = self.out_conv4(un7_4)

        cat_out = self.cat_out_conv(torch.cat((out1,out2 ,out3 , out4),dim=1))

        if not self.deepSupvision:
            return out1 , out2 , out3 , out4,cat_out
        else:
            return stage1Out1,stage1Out2,stage1Out3,stage1Out4,stage2Out1,stage2Out2,stage2Out3,stage2Out4,out1 , out2 , out3 , out4,cat_out
if __name__ == '__main__':
    x1 = torch.rand(1,1,128,128,128).cuda()
    x2 = torch.rand(1,1,128,128,128).cuda()
    x3 = torch.rand(1,1,128,128,128).cuda()
    x4 = torch.rand(1,1,128,128,128).cuda()
    mask = torch.rand(1,4).cuda()
    model = no_share_unet(in_channel=1, out_channel=3, diff=True,deepSupvision=True).cuda()
    res = model(x1,x2,x3,x4,mask)
