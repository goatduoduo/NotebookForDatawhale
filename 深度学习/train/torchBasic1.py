import torch

# 初始化一个0-11的数组
x = torch.arange(12)
print(x)
# 输出数组形状
print(x.shape)
# 输出数组大小
print(x.numel())
# 重塑为3*4数组
X = x.reshape(3, 4)
print(X)
# 新建一个2*3*4的数组
print(torch.zeros(2, 3, 4))
# 手动赋值数组
Y = torch.tensor([[2, 1, 3, 4], [1, 2, 3, 4], [5, 3, 2, 1]])
print(Y)
