
from SFusion import TF_RMBTS
#from RobustSeg import RobustSeg
#from UHVED import U_HVED
import torch
#from mmFormer.mmformer.mmformer import Model
#from m3ae.model.Unet import Unet_missing
#from ShaSpec.DualNet_SS import DualNet_SS

"""
### SFusion
model = TF_RMBTS(in_channels=1, out_channels=4, levels=4, feature_maps=16)
print(model)
dummy_input = torch.randn(1, 4, 128, 128, 128)
mask = torch.tensor([True, True, True, True]).unsqueeze(0).cuda()
model(dummy_input, mask)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters SFusion: {total_params}")
"""

### RobustSeg
model = RobustSeg(num_cls=4)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters RobustSeg: {total_params}")

"""
### UHVED
model = U_HVED(num_classes=4)
#print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters UHVED: {total_params}")


### mmFormer
model = Model(num_cls=4)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters mmFormer: {total_params}")

### m3ae
# pre-training
model = Unet_missing(input_shape = [128,128,128], pre_train = True, mask_ratio = 0.875, mdp = 3) 
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters pre-training m3ae: {total_params}")

# training
model = Unet_missing(input_shape = [128,128,128], init_channels = 16, out_channels=3, mdp=3, pre_train = False, deep_supervised = True, patch_shape = 128)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters training m3ae: {total_params}")

### ShaSpec
model = DualNet_SS(norm_cfg='IN', activation_cfg='LeakyReLU', num_classes=3, weight_std=True, self_att=False, cross_att=False)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters ShaSpec: {total_params}")
"""


