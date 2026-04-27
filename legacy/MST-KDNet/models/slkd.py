import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self):
        super(DistillKL, self).__init__()

    def forward(self, y_s, y_t, temp):
        T = temp.cuda()
        KD_loss = 0
        KD_loss += nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y_s/T, dim=1),
                                                        F.softmax(y_t/T, dim=1)) * T * T
        return KD_loss

class DistillKL_logit_stand(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self):
        super(DistillKL_logit_stand, self).__init__()

    def forward(self, y_s, y_t, temp):
        T = torch.tensor(temp, dtype=torch.float32).cuda() if isinstance(temp, int) else temp.cuda()
        KD_loss = 0
        KD_loss += nn.KLDivLoss(reduction='batchmean')(F.log_softmax(normalize(y_s)/T, dim=1),
                                                            F.softmax(normalize(y_t)/T, dim=1)) * T * T
        return KD_loss / y_s.numel()


if __name__ == '__main__':
    # 输入logit的大小为 (1, 4, 160, 192, 128)
    input_shape = (1, 4, 160, 192, 128)
    
    # 创建随机的logit张量，模拟学生和教师模型的输出
    y_s = torch.randn(input_shape).cuda()  # 学生模型的logit
    y_t = torch.randn(input_shape).cuda()  # 教师模型的logit
    
    # 假设温度值为3
    temp = 7
    
    # 创建DistillKL_logit_stand实例
    distill_model = DistillKL_logit_stand().cuda()
    
    # 计算知识蒸馏的损失
    kd_loss = distill_model(y_s, y_t, temp)
    
    # 输出损失值
    print(f"KD Loss: {kd_loss.item()}")