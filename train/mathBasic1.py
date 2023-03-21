import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A)
# 转置矩阵
print(A.T)

X = torch.arange(24).reshape(2, 3, 4)
# print(X)

# 相同形状的张量在进行二元运算后的结果也是相同形状的张量
B = A.clone()

print(A, A + B)

# 两个矩阵的元素乘法被称为“哈达玛积”
print(A * B)
print(2 + A)
print(2 * A)
# print(2 + X)
# print((2 * X).shape)
# 求和
print(233)
print(X)
print(X.sum())
# 对维度求和,也就是沿特定轴进行压扁求和~
print(X.sum(axis=1))
print(X.sum(axis=2))
print(X.sum(axis=[0, 1]))

# 保持维数不变
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)
print(A / sum_A)

# 矩阵向量乘法
x = torch.arange(4, dtype=torch.float32)
print(torch.mv(A, x))

# 矩阵乘法
B = torch.ones(4, 3)
print(torch.mm(A, B))

# 范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))

# L1范数，向量元素的绝对值求和
print(torch.abs(u).sum())

# F范数（弗罗贝尼乌斯范数），矩阵元素的平方和的平方根
print(torch.norm(torch.ones(4, 9)))
