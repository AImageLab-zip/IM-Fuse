import tensorly as tl
from tensorly.decomposition import parafac
import numpy as np

# 创建一个随机的五维张量，形状为 (1, 16, 40, 48, 36)
tensor = tl.tensor(np.random.rand(1, 16, 40, 48, 36))

# 设置分解的秩
rank = 5

# 进行 CP 分解
factors = parafac(tensor, rank=rank)

# factors 是一个包含各维度因子矩阵的列表
print(factors)

