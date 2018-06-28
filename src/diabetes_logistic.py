import numpy as np
import torch
from torch.autograd import Variable

xy = np.loadtxt('../data/diabetes.csv.gz', delimiter = ',', dtype = np.float32)
x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
y_data = Variable(torch.from_numpy(xy[-1]))

print(x_data.data.shape)
print(y_data.data.shape)