import torch
checkpoint_path = '/work/grana_neuro/missing_modalities/lckd_missing/last.pth'
checkpoint = torch.load(checkpoint_path,weights_only=False)
model = checkpoint['model']
model_state_dict = model.state_dict()
checkpoint['model']=model_state_dict
torch.save(checkpoint,checkpoint_path)