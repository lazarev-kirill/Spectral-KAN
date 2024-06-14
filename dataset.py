import pandas as pd
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor


class Data:
    def __init__(self, input, labels) -> None:
        self.input = torch.from_numpy(input)
        self.labels = torch.from_numpy(labels)
        self.len = len(labels)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.input[index], self.labels[index]


healthy = pd.read_excel("./good.xlsx")
unhealthy = pd.read_excel("./bad.xlsx")

healthy = healthy.iloc[:, 1:]
unhealthy = unhealthy.iloc[:, 1:]

healthy = healthy.to_numpy()
unhealthy = unhealthy.to_numpy()

labels_healthy = np.zeros(healthy.shape[1])
labels_unhealthy = np.ones(unhealthy.shape[1])

split = 20

h_train = healthy[:, :split]
h_test = healthy[:, split:]
u_train = unhealthy[:, :split]
u_test = unhealthy[:, split:]

labels_healthy_train = labels_healthy[:split]
labels_healthy_test = labels_healthy[split:]
labels_unhealthy_train = labels_unhealthy[:split]
labels_unhealthy_test = labels_unhealthy[split:]

X_train = np.hstack([h_train, u_train]).transpose()
X_test = np.hstack([h_test, u_test]).transpose()
y_train = np.hstack([labels_healthy_train, labels_unhealthy_train])
y_test = np.hstack([labels_healthy_test, labels_unhealthy_test])

train_dataset = Data(X_train, y_train)
test_dataset = Data(X_test, y_test)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=35)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=10)
