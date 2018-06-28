import numpy as np
import torch
from torch.autograd import Variable

xy = np.loadtxt('../data/diabetes.csv.gz', delimiter = ',', dtype = np.float32)
x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
y_data = Variable(torch.from_numpy(xy[:, [-1]]))

print(x_data.data.shape)
print(y_data.data.shape)

class Model(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 12)
        self.l2 = torch.nn.Linear(12, 10)
        self.l3 = torch.nn.Linear(10, 6)
        self.l4 = torch.nn.Linear(6, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        In the fowward function we accept a Variable of input data and we must
        return a Variable of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        out3 = self.sigmoid(self.l3(out2))
        y_pred = self.sigmoid(self.l4(out3))
        return y_pred

    
# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# train loop
for epoch in range(100):
    y_pred = model(x_data)

    # compute and print loss 
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])

    # zero gradients, perform a backward pass, and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

