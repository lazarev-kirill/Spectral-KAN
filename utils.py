import sys
from time import time

from numpy import mean

import torch.nn.functional as f

from torch import device, max, no_grad, Tensor, LongTensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pynvml import *

nvmlInit()

loss_function = {"cross_entropy": f.cross_entropy, "mse": f.mse_loss}


def get_loss(
    model: Module, valid_loader: DataLoader, device: device, loss_function_name: str
) -> float:
    with no_grad():
        mean_loss = []
        for x, y in valid_loader:
            x: Tensor
            y: Tensor
            x = x.to(device)
            y = y.to(device)
            y = y.long()
            y_pred = model(x)
            loss = loss_function[loss_function_name](y_pred, y)
            mean_loss.append(loss.item())
        mean_loss = sum(mean_loss) / len(mean_loss)
    return mean_loss


def get_accuracy(model: Module, valid_loader: DataLoader, device: device) -> float:
    model.eval()
    total = 0.0
    accuracy = 0.0
    with no_grad():
        for x, y in valid_loader:
            x: Tensor
            y: Tensor
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            _, prediction = max(y_pred, 1)
            total += y.size(dim=0)
            accuracy += (prediction == y).sum().item()
    accuracy = accuracy / total
    return accuracy


def train(
    model: Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    loss_function_name: str,
    num_of_epochs: int,
    device: device,
    logging: bool = True,
):
    print("Starting train...")
    s = time()
    mean_time_by_epoch = []
    epochs = range(num_of_epochs)
    train_losses = []
    valid_losses = []
    train_accuracy = []
    valid_accuracy = []
    mean_mem_alloc = []
    for epoch in epochs:
        s_epoch = time()
        current_train_loss = []
        model.train()
        for x, y in train_loader:
            x: Tensor
            y: Tensor
            x = x.to(device)
            y = y.to(device)
            y = y.long()
            y_pred = model(x)
            optimizer.zero_grad()
            loss = loss_function[loss_function_name](y_pred, y)
            loss.backward()
            optimizer.step()
            current_train_loss.append(loss.item())
            h = nvmlDeviceGetHandleByIndex(0)
            mean_mem_alloc.append(nvmlDeviceGetMemoryInfo(h).used / 1024 / 1024)
        e_epoch = time()
        mean_time_by_epoch.append(e_epoch - s_epoch)

        current_train_loss = sum(current_train_loss) / len(current_train_loss)
        current_valid_loss = get_loss(model, valid_loader, device, loss_function_name)

        current_train_acc = get_accuracy(model, train_loader, device)
        current_valid_acc = get_accuracy(model, valid_loader, device)

        train_losses.append(current_train_loss)
        valid_losses.append(current_valid_loss)
        train_accuracy.append(current_train_acc)
        valid_accuracy.append(current_valid_acc)

        if logging:
            sys.stdout.write(
                f"\rEpoch: {epoch + 1} | train loss: {round(current_train_loss, 6)} | train acc: {round(current_train_acc, 6)} | valid loss: {round(current_valid_loss, 6)} | valid acc: {round(current_valid_acc, 6)}"
            )
    e = time()
    print()
    total_time = e - s
    time_results = {
        "total": total_time,
        "mean_epoch": mean(mean_time_by_epoch),
        "plot": mean_time_by_epoch,
    }
    print("Mean memory allocated:", mean(mean_mem_alloc), "Mb")
    return train_losses, valid_losses, train_accuracy, valid_accuracy, time_results
