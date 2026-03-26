import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.nn as nn
from tqdm import tqdm

class DummyDataset(Dataset):
    def __init__(self, datapath:str|Path,splitfile:str|Path):
        self.datapath = Path(datapath)
        self.splitfile = Path(datapath)

    def __getitem__(self,idx:int):
        return {'image':torch.zeros((4,240,240,155)),
                'label':torch.zeros((240,240,155))}
    def __len__(self):
        return 5
    
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
    def forward(self, x:torch.Tensor, mask):
        return torch.rand_like(x) + self.dummy_param
    
if __name__ == '__main__':
    dataset = DummyDataset('no','path')
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=1)
    model = DummyModel()
    for element in tqdm(dataloader,total=len(dataloader)):
        out = model(element)
    torch.save(model.state_dict(),'/homes/ocarpentiero/IM-Fuse/templates/dummies/checkpoint.pth')
    