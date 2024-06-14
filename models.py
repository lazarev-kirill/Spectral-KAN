import torch
from torch import nn

from kan import KAN


class ModelSoftmax(nn.Module):
    def __init__(self) -> None:
        super(ModelSoftmax, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2000, out_features=1024),
            nn.SiLU(),
            nn.Linear(in_features=1024, out_features=2),
            nn.Softmax(dim=0),
        )

    def forward(self, x: torch.Tensor):
        out = self.layers(x.float())
        return out


class ModelSoftmaxExtended(nn.Module):
    def __init__(self) -> None:
        super(ModelSoftmaxExtended, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2000, out_features=1024),
            nn.SiLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.SiLU(),
            nn.Linear(in_features=1024, out_features=2),
            nn.Softmax(dim=0),
        )

    def forward(self, x: torch.Tensor):
        out = self.layers(x.float())
        return out


class ModelNoSoftmax(nn.Module):
    def __init__(self) -> None:
        super(ModelNoSoftmax, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2000, out_features=1024),
            nn.SiLU(),
            nn.Linear(in_features=1024, out_features=10),
        )

    def forward(self, x: torch.Tensor):
        out = self.layers(x.float())
        return out


class ModelNoSoftmaxExtended(nn.Module):
    def __init__(self) -> None:
        super(ModelNoSoftmaxExtended, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2000, out_features=1024),
            nn.SiLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.SiLU(),
            nn.Linear(in_features=1024, out_features=2),
        )

    def forward(self, x: torch.Tensor):
        out = self.layers(x.float())
        return out


class ModelSoftmax_KAN(nn.Module):
    def __init__(self) -> None:
        super(ModelSoftmax_KAN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2000, out_features=1024),
            nn.SiLU(),
            nn.Linear(in_features=1024, out_features=10),
            nn.SiLU(),
        )
        self.kan = KAN([10, 2], grid=10, device="cuda:0")
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor):
        out = self.layers(x.float())
        out = self.kan(out)
        out = self.softmax(out)
        return out


class ModelSoftmaxExtended_KAN(nn.Module):
    def __init__(self) -> None:
        super(ModelSoftmaxExtended_KAN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2000, out_features=1024),
            nn.SiLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.SiLU(),
            nn.Linear(in_features=1024, out_features=10),
            nn.SiLU(),
        )
        self.kan = KAN([10, 2], grid=10, device="cuda:0")
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor):
        out = self.layers(x.float())
        out = self.kan(out)
        out = self.softmax(out)
        return out


class ModelNoSoftmax_KAN(nn.Module):
    def __init__(self) -> None:
        super(ModelNoSoftmax_KAN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2000, out_features=1024),
            nn.SiLU(),
            nn.Linear(in_features=1024, out_features=10),
            nn.SiLU(),
        )
        self.kan = KAN([10, 2], grid=10, device="cuda:0")

    def forward(self, x: torch.Tensor):
        out = self.layers(x.float())
        out = self.kan(out)
        return out


class ModelNoSoftmaxExtended_KAN(nn.Module):
    def __init__(self) -> None:
        super(ModelNoSoftmaxExtended_KAN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2000, out_features=1024),
            nn.SiLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.SiLU(),
            nn.Linear(in_features=1024, out_features=10),
            nn.SiLU(),
        )
        self.kan = KAN([10, 2], grid=10, device="cuda:0")

    def forward(self, x: torch.Tensor):
        out = self.layers(x.float())
        out = self.kan(out)
        return out
