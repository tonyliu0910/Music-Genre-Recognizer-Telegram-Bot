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
from efficientnet_pytorch import EfficientNet

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

"""# Dataloader"""

# data_dir = './2022-adar-ai-training-final/CIFAR100'
# data_dir = ""
# classes = os.listdir(data_dir + "Images/")
# print(classes)
# print(len(classes))



# # PyTorch datasets
# train_ds = ImageFolder(data_dir + "Images/", train_tfms)
# valid_ds = ImageFolder(data_dir + "Images2/", valid_tfms)

# # PyTorch data loaders
# batch_size = 16
# train_dl = DataLoader(train_ds, batch_size, shuffle=True)
# valid_dl = DataLoader(valid_ds, 256)


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


# train_dl = DeviceDataLoader(train_dl, device)
# valid_dl = DeviceDataLoader(valid_dl, device)

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




def load_pretrained_weights(model, model_name, weights_path=None, load_fc=True, advprop=False, verbose=True):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): The whole model of efficientnet.
        model_name (str): Model name of efficientnet.
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        advprop (bool): Whether to load pretrained weights
                        trained with advprop (valid when weights_path is None).
    """
    if isinstance(weights_path, str):
        state_dict = torch.load(weights_path)
    else:
        # AutoAugment or Advprop (different preprocessing)
        url_map_ = url_map_advprop if advprop else url_map
        state_dict = model_zoo.load_url(url_map_[model_name])

    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        assert not ret.missing_keys, 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        ret = model.load_state_dict(state_dict, strict=False)
        assert set(ret.missing_keys) == set(
            ['_fc.weight', '_fc.bias']), 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    assert not ret.unexpected_keys, 'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)

    if verbose:
        print('Loaded pretrained weights for {}'.format(model_name))






class EfficientNetB0_cifar100(ImageClassificationBase):
    def __init__(self, efnb0):
        super(EfficientNetB0_cifar100, self).__init__()
        self.efnb0 = efnb0

    def forward(self, input):
        out = self.efnb0(input)
        return out



# state = torch.load('tresnet_m.pth', map_location='cpu')['model']
# model.load_state_dict(state, strict=False)

"""# Set Config"""

# epochs = 300
# max_lr = 0.0001
# opt_func = torch.optim.Adam(model.parameters(), lr=max_lr)  # like Adam, SGD
# scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_func, milestones=[5], gamma=0.1)
# lrs = []

"""# Training

"""

from torch.optim import optimizer


def Train(epochs, max_lr, model, train_loader, val_loader, opt_func):
    torch.cuda.empty_cache()
    history = []
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

# history = []
# history += Train(epochs, max_lr, model, train_dl, valid_dl, opt_func)

"""# Model / Parameter statistics"""

# Print model
# print(model)

# Print parameter
# parameter_out = open("EfficientNet_parameters.txt", "a")
# parameter_out.write(summary(model, torch.rand((1, 3, 32, 32)).to(device)))
# parameter_out.close()
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


# plot_losses(history)

"""# Testing"""


def test(model, valid_dl):
    with torch.no_grad():
        model.eval()
        valid_loss = 0
        correct = 0
        bs = 256
        result = []
        check_names = []
        for i, (data, target) in enumerate(valid_dl):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            arr = pred.data.cpu().numpy()
            for j in range(pred.size()[0]):
                file_name = valid_ds.samples[i * bs + j][0].split("/")[-1]
                result.append((file_name, pred[j].cpu().numpy()[0]))

    return result


# PATH = "EfficientNet-b0.pth"
# torch.save(model.state_dict(), PATH)

# """# Save Result"""

# with open("EfficientNet-B0_result.csv", "w") as f:
#     f.write("Id,Category\n")
#     for data in result:
#         f.write(data[0] + '," ' + str(data[1]) + '"\n')