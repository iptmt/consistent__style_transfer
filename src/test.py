import torch

a = torch.randn(1, 5)
b = torch.randn(5, 10)

c = a.mm(b).reshape(-1)
print(c.shape)
d = (a.squeeze(0).unsqueeze(-1) * b).sum(0)
print(d.shape)
print(c - d)