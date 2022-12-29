import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder, CIFAR100
from torchtoolbox.tools import summary
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import re
import math
import collections
from functools import partial
from torch.utils import model_zoo


# %matplotlib inline

matplotlib.rcParams["figure.facecolor"] = "#ffffff"

"""# Check GPU"""


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


device = get_default_device()
print(device)

"""# Dataloader"""

# data_dir = './2022-adar-ai-training-final/CIFAR100'
data_dir = "dataset/"
classes = os.listdir(data_dir + "Images/")
print(classes)
print(len(classes))

# Data transforms (normalization & data augmentation)
stats = ((0.4306, 0.4185, 0.3943), (0.2938, 0.2855, 0.2902))
train_tfms = tt.Compose(
    [
        # =============================================
        #  tt.Resize(224),
        tt.ToTensor(),
        #  tt.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1944, 0.2010), inplace=True)
        # =============================================
    ]
)

valid_tfms = tt.Compose(
    [
        # =============================================
        #  tt.Resize(224),
        tt.ToTensor(),
        #  tt.Normalize( mean = (0.4914, 0.4822, 0.4465),std = (0.2023, 0.1944, 0.2010))
        # =============================================
    ]
)

# PyTorch datasets
whole_ds = ImageFolder(data_dir + "Images/", train_tfms)
train_ds, valid_ds = torch.utils.data.random_split(
    whole_ds, [int(len(whole_ds) * 0.8), len(whole_ds) - int(len(whole_ds) * 0.8)]
)

# PyTorch data loaders
batch_size = 64
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, 256)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)

"""# Model"""


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def conv_block(
    in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=False
):
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
    ]
    if pool:
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        # =============================================
        data, label = batch
        prediction = self.forward(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(prediction, label)
        # =============================================
        return loss

    def validation_step(self, batch):
        # =============================================
        data, label = batch
        prediction = self(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(prediction, label)
        acc = accuracy(prediction, label)
        # =============================================
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies

        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch,
                result["lrs"][-1],
                result["train_loss"],
                result["val_loss"],
                result["val_acc"],
            )
        )


from efficientnet_pytorch import EfficientNet

efnb0 = EfficientNet.from_pretrained("efficientnet-b0")

# for param in efnb0.parameters():
#   param.requires_grad = False

efnb0._fc = nn.Sequential(
    nn.Linear(in_features=1280, out_features=320),
    nn.ReLU(),
    nn.Linear(in_features=320, out_features=80),
    nn.ReLU(),
    nn.Linear(in_features=80, out_features=15),
)


class EfficientNetB0_cifar100(ImageClassificationBase):
    def __init__(self):
        super(EfficientNetB0_cifar100, self).__init__()
        self.efnb0 = efnb0

    def forward(self, input):
        out = self.efnb0(input)
        return out


model = to_device(EfficientNetB0_cifar100(), device)
# state = torch.load('tresnet_m.pth', map_location='cpu')['model']
# model.load_state_dict(state, strict=False)

"""# Set Config"""

epochs = 30
max_lr = 0.0001
# opt_func = torch.optim.SGD(
#     model.parameters(), lr=max_lr, momentum=0.9, weight_decay=0.0005, nesterov=True
# )
opt_func = torch.optim.Adam(model.parameters(), lr=max_lr)  # like Adam, SGD
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_func, T_max=epochs)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_func, milestones=[5], gamma=0.1)
lrs = []

"""# Training

"""

from torch.optim import optimizer


def Train(epochs, max_lr, model, train_loader, val_loader, opt_func):
    torch.cuda.empty_cache()
    history = []
    best_accuracy = 0
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            # =============================================
            opt_func.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            opt_func.step()
            train_losses.append(loss.item())
            # =============================================
        # Validation phase
        result = evaluate(model, val_loader)
        if result["val_acc"] > best_accuracy:
            best_accuracy = result["val_acc"]
            torch.save(model.state_dict(), "best.pth")
        # =============================================
        t_loss = sum(train_losses) / len(train_losses)
        result["train_loss"] = t_loss
        lrs.append(opt_func.param_groups[0]["lr"])
        scheduler.step()
        result["lrs"] = lrs
        # =============================================
        model.epoch_end(epoch, result)
        history.append(result)
    return history


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# Commented out IPython magic to ensure Python compatibility.

history = []
history += Train(epochs, max_lr, model, train_dl, valid_dl, opt_func)

"""# Model / Parameter statistics"""

# Print model
print(model)

# Print parameter
parameter_out = open("EfficientNet_parameters.txt", "a")
parameter_out.write(summary(model, torch.rand((1, 3, 32, 32)).to(device)))
parameter_out.close()
"""# Plot Learning curve"""


def plot_losses(history):
    val_losses = [x["val_loss"] for x in history]
    train_losses = [x["train_loss"] for x in history]
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(val_losses, label="val")
    plt.plot(train_losses, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("Efficient_loss.png")


plot_losses(history)

"""# Testing"""


# def test(model, valid_dl):
#     with torch.no_grad():
#         model.eval()
#         valid_loss = 0
#         correct = 0
#         bs = 256
#         result = []
#         check_names = []
#         for i, (data, target) in enumerate(valid_dl):
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             pred = output.data.max(1, keepdim=True)[
#                 1
#             ]  # get the index of the max log-probability
#             arr = pred.data.cpu().numpy()
#             for j in range(pred.size()[0]):
#                 file_name = valid_ds.samples[i * bs + j][0].split("/")[-1]
#                 result.append((file_name, pred[j].cpu().numpy()[0]))

#     return result


# result = test(model, valid_dl)
# PATH = "EfficientNet-b0.pth"
torch.save(model.state_dict(), "last.pth")

"""# Save Result"""

# with open("EfficientNet-B0_result.csv", "w") as f:
#     f.write("Id,Category\n")
#     for data in result:
#         f.write(data[0] + '," ' + str(data[1]) + '"\n')
