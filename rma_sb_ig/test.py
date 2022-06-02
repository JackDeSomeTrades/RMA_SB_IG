import torch
import numpy as np
l = np.array([1,2,3,4])
num_env = 5

real_ids = np.array([1,2,8,9])
a = torch.rand(3,4)

c = torch.rand(3,1)
print(a)
print(c)
b = torch.zeros(3,10)
b[:,real_ids] = c
print(b)