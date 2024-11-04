import torch

a = torch.randn((1,28,28))
b=a.view([1,])

print(a.shape)
print(b.shape)