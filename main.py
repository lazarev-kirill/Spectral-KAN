import os
import matplotlib.pyplot as plt
import torch
import numpy as np

from kan import KAN

from utils import train
from dataset import train_loader, test_loader

device = torch.device("cuda:0")

epochs = 3000

repeats = 1

err_losses_train = []
err_losses_valid = []
err_accs_train = []
err_accs_valid = []

for i in range(repeats):
    model = KAN([2000, 2], grid=5, device="cuda:0", symbolic_enabled=False)
    optimizer = torch.optim.Adam(model.parameters(), 1e-6)
    train_losses, valid_losses, train_accuracy, valid_accuracy, time_results = train(
        model,
        optimizer,
        train_loader,
        test_loader,
        "cross_entropy",
        epochs,
        device,
    )
    err_losses_train.append(train_losses)
    err_losses_valid.append(valid_losses)
    err_accs_train.append(train_accuracy)
    err_accs_valid.append(valid_accuracy)

err_losses_train = np.vstack(err_losses_train)
err_losses_valid = np.vstack(err_losses_valid)
err_accs_train = np.vstack(err_accs_train)
err_accs_valid = np.vstack(err_accs_valid)

mean_err_losses_train = []
mean_err_losses_valid = []
mean_err_accs_train = []
mean_err_accs_valid = []
std_err_losses_train = []
std_err_losses_valid = []
std_err_accs_train = []
std_err_accs_valid = []
for i in range(epochs):
    mean_err_losses_train.append(np.mean(err_losses_train[:, i]))
    mean_err_losses_valid.append(np.mean(err_losses_valid[:, i]))
    mean_err_accs_train.append(np.mean(err_accs_train[:, i]))
    mean_err_accs_valid.append(np.mean(err_accs_valid[:, i]))

    std_err_losses_train.append(np.std(err_losses_train[:, i]))
    std_err_losses_valid.append(np.std(err_losses_valid[:, i]))
    std_err_accs_train.append(np.std(err_accs_train[:, i]))
    std_err_accs_valid.append(np.std(err_accs_valid[:, i]))

if not os.path.isdir(model._get_name()):
    os.mkdir(model._get_name())
    with open(f"{model._get_name()}/{model._get_name()}_description.txt", "w") as f:
        f.write(model.__str__())

plt.title(f"Losses {model._get_name()}")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.errorbar(
    range(epochs), mean_err_losses_train, yerr=std_err_losses_train, label="train"
)
plt.errorbar(
    range(epochs), mean_err_losses_valid, yerr=std_err_losses_valid, label="valid"
)
plt.legend()
plt.savefig(f"{model._get_name()}/losses.png")
plt.close("all")

plt.title(f"Accuracy {model._get_name()}")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.errorbar(range(epochs), mean_err_accs_train, yerr=std_err_accs_train, label="train")
plt.errorbar(range(epochs), mean_err_accs_valid, yerr=std_err_accs_valid, label="valid")
plt.legend()
plt.savefig(f"{model._get_name()}/accuracy.png")
plt.close("all")

plt.plot(range(epochs), time_results["plot"])
plt.savefig(f"{model._get_name()}/time.png")
plt.close("all")

lib = ["x", "x^2", "x^3", "x^4", "exp", "log", "sqrt", "tanh", "sin", "abs"]
symbols = model.auto_symbolic(lib=lib)
with open("file.txt", "w") as file:
    file.write(symbols)

# model.plot("C:/KAN_images/f1", scale=25)
# plt.savefig(f"{model._get_name()}/main_image.pdf")
# plt.close("all")

# model2 = model.prune()
# model.prune()
# model.plot("C:/KAN_images/f2", scale=25, beta=1)
# plt.savefig(f"{model._get_name()}/prune.pdf")
# plt.close("all")

# for x, y in train_loader:
#     x: torch.Tensor
#     x = x.to(device)
#     model2(x)
#     break

# model2.plot("C:/KAN_images/f3", scale=25, beta=1)
# plt.savefig(f"{model._get_name()}/removed_neurons.pdf")
# plt.close("all")
