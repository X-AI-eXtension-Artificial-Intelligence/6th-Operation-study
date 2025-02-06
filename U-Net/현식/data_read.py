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

# GPU 설정하기
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(device)

# 데이터 경로
root_path = './cityscapes_data'
data_dir = root_path

# data_dir의 경로(문자열)와 train(문자열)을 결합해서 train_dir(train 폴더의 경로)에 저장
train_dir = os.path.join(data_dir, "train")

# data_dir의 경로(문자열)와 val(문자열)을 결합해서 val_dir(val 폴더의 경로)에 저장
val_dir = os.path.join(data_dir, "val")

# train_dir 경로에 있는 모든 파일을 리스트의 형태로 불러와서 train_fns에 저장
train_fns = os.listdir(train_dir)

# val_dir 경로에 있는 모든 파일을 리스트의 형태로 불러와서 val_fns에 저장
val_fns = os.listdir(val_dir)

print(len(train_fns), len(val_fns))


# 0~255 사이의 숫자를 3*num_items번 랜덤하게 뽑기
num_items = 1000
color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)
print(color_array.shape)


# K-means clustering 알고리즘을 사용하여 label_model에 저장 및 학습
num_classes = 10
label_model = KMeans(n_clusters = num_classes)
label_model.fit(color_array)