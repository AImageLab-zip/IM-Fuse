import torch
import torch.nn.functional as F

def compute_extreme_map(weights_list):
    """
    对输入的 4D 张量列表 (B, C, H, W) 进行如下处理：
      1) 堆叠后在第0维度计算 max/min/mean, 得到 (B, C, H, W)
      2) 分别与列表中每个张量做逐元素乘法

    返回:
      max_list, min_list, mean_list (三个列表, 每个元素形状与原张量一致 (B, C, H, W))
    """
    # 1) 将 N 个 (B, C, H, W) 张量在 0 维堆叠 -> (N, B, C, H, W)
    stacked = torch.stack(weights_list, dim=0)
    
    # 2) 在堆叠后维度0计算最大/最小/平均值 -> 形状均为 (B, C, H, W)
    max_map = stacked.max(dim=0).values
    min_map = stacked.min(dim=0).values
    mean_map = stacked.mean(dim=0)

    # 3) 分别与原列表中的每个权重张量做逐元素乘法
    max_list  = [W * max_map  for W in weights_list]
    min_list  = [W * min_map  for W in weights_list]
    mean_list = [W * mean_map for W in weights_list]

    return max_list, min_list, mean_list

def EVD(teacher_extract_weights, student_extract_weights):
    """
    Extreme Value Distillation主要函数：对教师 / 学生各自的extract_weights列表 (每个元素形状 (B, C, H, W)) 执行
    同样的【取极值映射+逐元素乘法】操作, 然后用 MSE 方式进行蒸馏, 最终返回总的 MSE 损失。

    :param teacher_extract_weights: 教师模型的权重列表, [ (B, C, H, W), (B, C, H, W), ... ]
    :param student_extract_weights: 学生模型的权重列表, [ (B, C, H, W), (B, C, H, W), ... ]

    注意:
      - teacher_extract_weights 和 student_extract_weights 的长度应相同, 各元素的 (B, C, H, W) 也要能对应.
      - 若 B 不同, 需要考虑如何对齐或先将 batch 维合并/拆分等.
    """

    # 1) 分别计算：教师的 (max_list, min_list, mean_list)
    teacher_max_list, teacher_min_list, teacher_mean_list = compute_extreme_map(teacher_extract_weights)
    # 2) 分别计算：学生的 (max_list, min_list, mean_list)
    student_max_list, student_min_list, student_mean_list = compute_extreme_map(student_extract_weights)

    mse_loss = 0.0

    # 3) 对max_list进行MSE蒸馏
    for t_max, s_max in zip(teacher_max_list, student_max_list):
        mse_loss += F.mse_loss(t_max, s_max, reduction='mean')

    # 对min_list进行MSE蒸馏
    for t_min, s_min in zip(teacher_min_list, student_min_list):
        mse_loss += F.mse_loss(t_min, s_min, reduction='mean')

    # 对mean_list进行MSE蒸馏
    for t_mean, s_mean in zip(teacher_mean_list, student_mean_list):
        mse_loss += F.mse_loss(t_mean, s_mean, reduction='mean')

    # 简单平均，也可自行设定加权策略
    # total_len = len(teacher_max_list)
    mse_loss = mse_loss

    return mse_loss

# ------------------- 示例用法 -------------------
if __name__ == "__main__":

    teacher_extract_weights = [
        torch.randn(1, 12, 120, 120),
        torch.randn(1, 12, 120, 120),
        torch.randn(1, 12, 120, 120),
    ]
    student_extract_weights = [
        torch.randn(1, 12, 120, 120),
        torch.randn(1, 12, 120, 120),
        torch.randn(1, 12, 120, 120),
    ]

    # 计算 MSE 蒸馏损失
    mse_loss_value = EVD(
        teacher_extract_weights,
        student_extract_weights
    )
    print("MSE Distillation Loss =", mse_loss_value.item())
