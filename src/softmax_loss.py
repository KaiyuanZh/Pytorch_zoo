import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# cross entropy example
# One-hot representation
# 0: 1 0 0
# 1: 0 1 0
# 2: 0 0 1
Y = np.array([1, 0, 0])

Y_pred1 = np.array([0.7, 0.2, 0.1]) # the sum of the three properties should be 1
Y_pred2 = np.array([0.1, 0.3, 0.6])
print("loss1 = ", np.sum(np.dot(-Y, np.log(Y_pred1))))
print("loss2 = ", np.sum(np.dot(-Y, np.log(Y_pred2))))


# Softmax + CrossEntropy (logSoftmax + NLLLoss)
loss = nn.CrossEntropyLoss()

# target is of size nBatch
# each element in target has to have 0<= value <= nClasses (0 - 2)
# Input is class, not one-hot
Y = Variable(torch.LongTensor([0]), requires_grad=False)

# input is of sieze nBatch x nClasses = 1 x 4
# Y_pred are logits (not softmax)
Y_pred1 = Variable(torch.tensor([[2.0, 1.0, 0.1]]))
Y_pred2 = Variable(torch.tensor([[0.5, 2.0, 0.3]]))

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print("pytorch loss1 = ", l1.data, "\npytorch loss2 = ", l2.data)

print("Y_pred1 = ", torch.max(Y_pred1.data, 1)[1])
print("Y_pred2 = ", torch.max(Y_pred2.data, 1)[1])

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot

Y = Variable(torch.LongTensor([2, 0, 1]), requires_grad=False)

# input is of size nBatch x nClasses = 2 x 4
# Y_pred are logits (not softmax)
Y_pred1 = Variable(torch.Tensor([[0.1, 0.2, 0.9],
                                 [1.1, 0.1, 0.2],
                                 [0.2, 2.1, 0.1]]))

Y_pred2 = Variable(torch.Tensor([[0.8, 0.2, 0.3],
                                 [0.2, 0.3, 0.5],
                                 [0.2, 0.2, 0.5]]))

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print("Batch loss1 = ", l1.data, "Batch loss2 = ", l2.data)