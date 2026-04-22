import torch
from nets.cph import CPH
class InputAdapter(torch.nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(k, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight[:] = 1.0 / k
    def forward(self, x):
        return self.conv(x)
    
net = torch.nn.Sequential(InputAdapter(4), CPH(n_classes=3)).to('cuda')
ms = [int(50 * 0.15), int(50 * 0.35), int(50 * 0.55), int(50 * 0.7)]
def lr_lambda(epoch):
    if epoch < max(1, 5):
        return (epoch + 1) / max(1, 5)
    steps = sum(int(m <= epoch) for m in ms)
    return 0.1 ** steps


net = net.to(memory_format=torch.channels_last)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=4e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


checkpoint_path = '/work/grana_neuro/missing_modalities/ReHyDIL/checkpoints/model_CPH_last_t1n.pth'
checkpoint = torch.load(checkpoint_path,weights_only=False)
epochs = checkpoint['epoch']
for epoch in range(epochs):
    scheduler.step()
checkpoint['scheduler'] = scheduler.state_dict()
torch.save(checkpoint,checkpoint_path)