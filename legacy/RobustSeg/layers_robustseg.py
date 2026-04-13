from curses.ascii import alt
import torch.nn as nn
import torch.nn.functional as F
import torch

def normalize(planes,norm='in'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(32,planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes, eps=1e-06)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class general_conv3d(nn.Module):
    def __init__(self,in_ch,out_ch,k_size=3,stride=1, padding=1, pad_type='zeros', norm='in', drop_rate=0.0
                ,is_training=True, relufactor=0.2, act_type='lrelu'):
        super(general_conv3d,self).__init__()
        self.drop_rate = drop_rate
        self.is_training = is_training
        self.act_type = act_type
        self.norm = norm
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, 
                            stride=stride, padding=padding, padding_mode=pad_type, bias=True)
        if norm is not None:
            self.normalize = normalize(out_ch,norm)
        if act_type == 'lrelu':
            self.activation = nn.LeakyReLU(relufactor)
        elif act_type == 'relu':
            self.activation = nn.ReLU()

    def forward(self,input):
        conv = self.conv(input)
        conv = F.dropout(conv,p=self.drop_rate,training=self.is_training)
        if self.norm is not None:
            conv = self.normalize(conv)
        if self.act_type is not None:
            conv = self.activation(conv)
        return conv

class dilate_conv2d(nn.Module):
    def __init__(self,in_ch,out_ch,k_size=3,stride=1, padding=1,drop_rate=0.0, do_norm=True, do_relu=True
                , relufactor=0, norm_type=None,is_training=True):
        super(dilate_conv2d,self).__init__()
        self.drop_rate = drop_rate
        self.do_norm = do_norm
        self.do_relu = do_relu
        self.drop_rate = drop_rate
        self.relufactor = relufactor
        self.norm_type = norm_type
        self.is_training = is_training
        self.conv2d = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=k_size,stride=stride,padding=padding)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(relufactor)

    def forward(self,inputconv):
        di_conv_2d = self.conv2d(inputconv)
        if self.drop_rate != 0:
            di_conv_2d = F.dropout(di_conv_2d,p=self.drop_rate,training = self.is_training)
        if self.do_norm:
            if self.norm_type is None:
                print("normalization type is not specified!")
                exit(1)
            elif self.norm_type=='in':
                di_conv_2d = nn.InstanceNorm2d(di_conv_2d)
            elif self.norm_type=='bn':
                di_conv_2d = nn.BatchNorm2d(di_conv_2d)
        
        if self.do_relu:
            if(self.relufactor == 0):
                di_conv_2d = self.relu(di_conv_2d)
            else:
                di_conv_2d = self.lrelu(di_conv_2d)

        return di_conv_2d

class general_deconv2d(nn.Module):
    def __init__(self,in_ch,out_ch,k_size=3,stride=1, padding=1,drop_rate=0.0, do_norm=True, do_relu=True
                , relufactor=0, norm_type=None,is_training=True):
        super(general_deconv2d,self).__init__()
        self.drop_rate = drop_rate
        self.do_norm = do_norm
        self.do_relu = do_relu
        self.drop_rate = drop_rate
        self.relufactor = relufactor
        self.norm_type = norm_type
        self.is_training = is_training
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(relufactor)
        self.ConvTranspose2d = nn.ConvTranspose2d(in_channels=in_ch,out_channels=out_ch,kernel_size=k_size,
                                stride=stride,padding=padding)

    def forward(self,inputconv):
        conv = self.ConvTranspose2d(inputconv)
        if self.do_norm:
            if self.norm_type is None:
                print ("normalization type is not specified!")
                exit(1)
            elif self.norm_type=='in':
                di_conv_2d = nn.InstanceNorm2d(di_conv_2d)
            elif self.norm_type=='bn':
                di_conv_2d = nn.BatchNorm2d(di_conv_2d)

        if self.do_relu:
            if(self.relufactor == 0):
                conv = self.relu(conv)
            else:
                conv = self.lrelu(conv)
        return conv

class linear(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(linear,self).__init__()
        self.linear = nn.Linear(in_features=in_ch,out_features=out_ch,bias=True)

    def forward(self,x):
        x = x.view(x.shape[0],-1)
        x = self.linear(x)
        return x

class adaptive_resblock(nn.Module):
    def __init__(self):
        super(adaptive_resblock,self).__init__()
        self.conv1 = general_conv3d(in_ch=128, out_ch=128, k_size=3, stride=1, padding=1, pad_type='reflect', norm=None, act_type=None)
        self.conv2 = general_conv3d(in_ch=128, out_ch=128, k_size=3, stride=1, padding=1, pad_type='reflect', norm=None,act_type=None)
        self.relu = nn.ReLU()

    def forward(self, x_init, channels, mu, sigma): #x:[B,128,10,10,10] mu:[B,128,1,1,1] sigma[B,128,1,1,1]
        x = self.conv1(x_init)                  #[B,128,10,10,10]              
        x = adaptive_instance_norm(x, mu, sigma)  #[B,128,10,10,10]
        x = self.relu(x)                        #[B,128,10,10,10]

        x = self.conv2(x)                       #[B,128,10,10,10]
        x = adaptive_instance_norm(x, mu, sigma)  #[B,128,10,10,10]

        return x + x_init                       #[B,128,10,10,10]

def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):    #content:[B,128,10,10,10] gamma:[B,128,1,1,1] beta:[B,128,1,1,1]
        c_mean = torch.mean(content, axis=[2,3,4], keepdims=True) #[B,128,1,1,1]
        c_var = torch.var(content,axis=[2,3,4], keepdims=True)   #[B,128,1,1,1]
        c_std = torch.sqrt(c_var + epsilon)                     #[B,128,1,1,1]
        return gamma * ((content - c_mean) / c_std) + beta

def up_sample(x, scale_factor=2):
    up = nn.Upsample(size=None, scale_factor=scale_factor, mode='nearest', align_corners=None)
    return up(x)
