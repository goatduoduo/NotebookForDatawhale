import torch

# 广播机制会用前面的元素去复制不存在的列
a = torch.arange(3).reshape((3, 1))
b = torch.arange(3).reshape((1, 3))
print(a, b)
print(a + b)

x = torch.arange(12).reshape((3, 4))
print(x)
x[0:2, :] = 12
print(x)

# 使用x[:] = x+y 或者 x += y减少内存开销
