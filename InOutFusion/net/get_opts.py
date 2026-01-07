import os
import sys
sys.path.insert(0, '/Code/SFusion-main/')


from affnet.models import arguments_model
import argparse
parser = argparse.ArgumentParser(description="Training arguments", add_help=True)

from affnet import modeling_arguments
parser = modeling_arguments(parser=parser)
from options.utils import load_config_file
opts = parser.parse_args()
# print(opts)

from affnet.modules.aff_block import AFNO3D_channelfirst




block = AFNO3D_channelfirst(opts, hidden_size=128)

import torch
x = torch.rand((4,128,8,8, 8))
y = block(x)
print(y.size())


from net.BasicBlock import TF_3D

x = x.cuda()
tf_block = TF_3D(128, trans='ff').cuda()

y2 = tf_block([x,x])
print(y2.size())