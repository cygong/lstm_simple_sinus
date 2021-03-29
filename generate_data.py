import numpy as np
import matplotlib.pyplot as plt
import torch

#dataset
from torch.utils.data import Dataset

class timeseries(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

    def __len__(self):
        return self.len

def generate_data():
    #creating the dataset
    x = np.arange(1,721,1)
    y = np.sin(x*np.pi/180) + np.random.randn(720)*0.05

    # structuring the data
    X = []
    Y = []
    for i in range(0,710):
        list1 = []
        for j in range(i,i+10):
            list1.append(y[j+1])
        X.append(list1)
        Y.append(y[j])

    #train test split
    X = np.array(X)
    Y = np.array(Y)
    x_train = X[:360]
    x_test = X[360:]
    y_train = Y[:360]
    y_test = Y[360:]

    dataset_train = timeseries(x_train,y_train)
    dataset_test = timeseries(x_test,y_test)

    return dataset_train,dataset_test

