import torch

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)

x1 = torch.arange(12, dtype=torch.float32).reshape((3, 4))
y1 = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# dim 0为按行，dim 1为按列，也就是按照哪个维度合并
print(torch.cat((x1, y1), dim=0))
print(torch.cat((x1, y1), dim=1))
# 为你的张量求和
print(x1.sum())
