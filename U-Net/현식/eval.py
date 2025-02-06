import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm.notebook import tqdm

from data_read import *
from dataset import *
from model import *
from train import *

model_ = UNet(num_classes=num_classes).to(device)
model_.load_state_dict(torch.load(model_save_path))

test_batch_size = 8
dataset = CityscapeDataset(val_dir, label_model)
data_loader = DataLoader(dataset, batch_size=test_batch_size)

X, Y = next(iter(data_loader))
X, Y = X.to(device), Y.to(device)
Y_pred = model_(X)
print(Y_pred.shape)
Y_pred = torch.argmax(Y_pred, dim=1)
print(Y_pred.shape)

torch.Size([8, 64, 256, 256])
torch.Size([8, 256, 256])
inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
])

save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)

fig, axes = plt.subplots(test_batch_size, 3, figsize=(3*5, test_batch_size*5))

iou_scores = []

for i in range(test_batch_size):
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    landscape = (landscape * 255).astype(np.uint8)
    Image.fromarray(landscape).save(os.path.join(save_dir, f"sample_{i}_landscape.png"))

    label_class = Y[i].cpu().detach().numpy().astype(np.uint8)
    Image.fromarray(label_class * 255).save(os.path.join(save_dir, f"sample_{i}_label.png"))

    label_class_predicted = Y_pred[i].cpu().detach().numpy().astype(np.uint8)
    Image.fromarray(label_class_predicted * 255).save(os.path.join(save_dir, f"sample_{i}_predicted.png"))

    intersection = np.logical_and(label_class, label_class_predicted)
    union = np.logical_or(label_class, label_class_predicted)
    iou_score = np.sum(intersection) / np.sum(union)
    iou_scores.append(iou_score)

    axes[i, 0].imshow(landscape)
    axes[i, 0].set_title("Landscape")
    axes[i, 1].imshow(label_class)
    axes[i, 1].set_title("Label Class")
    axes[i, 2].imshow(label_class_predicted)
    axes[i, 2].set_title("Label Class - Predicted")

plt.show()