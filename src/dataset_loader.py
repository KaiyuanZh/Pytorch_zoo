# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class DiabetesDataset(Dataset):
    """
    Example: Diabetes dataset
    """
    
    # initialize the data, download, read data, etc
    def __init__(self):
        xy = np.loadtxt('../data/diabetes.csv.gz',
                        delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    # return one item on the index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    # return the data length
    def __len__(self):
        return self.len
    

dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):

        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # run the training process
        print(epoch, i, "inputs", inputs.data, "labels", labels.data)