
import torch
import numpy as np

t1 = torch.FloatTensor([1, 2, 3])
t2 = torch.stack((t1, t1 * 2))

print(t2)
