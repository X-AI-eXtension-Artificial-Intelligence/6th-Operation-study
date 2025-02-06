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

class CityscapeDataset(Dataset):
  def __init__(self, image_dir, label_model):
    # image_dir : 이미지 파일들이 저장된 디렉토리 경로
    # label_model : 이미지에 있는 객체의 군집을 예측하기 위해 사용되는 K-means 군집화 모델
    self.image_dir = image_dir
    self.image_fns = os.listdir(image_dir) #지정된 디렉토리 내의 파일 목록 가져와 저장
    self.label_model = label_model
    
  def __len__(self) :
    return len(self.image_fns) #디렉토리 내 이미지 파일 수 반환
    
  def __getitem__(self, index) :
    # 지정된 인덱스의 이미지 파일 이름을 가져와, 전체 파일 경로를 구성
    # 해당 파일 경로에서 이미지를 열고, 이를 Numpy 배열로 변환
    image_fn = self.image_fns[index]
    image_fp = os.path.join(self.image_dir, image_fn)
    image = Image.open(image_fp)
    image = np.array(image)
    #이미지를 두 부분(원본 이미지와 레이블 이미지)로 분리
    cityscape, label = self.split_image(image)
    #군집화를 통해 얻은 정보(각 픽셀이 속한 군집의 정보)를 원래 이미지 형태로 다시 만듦 
    label_class = self.label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
    #군집화 결과를 텐서로 변환
    label_class = torch.Tensor(label_class).long()
    #원본 이미지를 변환(정규화,텐서변환)
    cityscape = self.transform(cityscape)
    return cityscape, label_class
    
  def split_image(self, image) :
    image = np.array(image)
    cityscape, label = image[ : , :256, : ], image[ : , 256: , : ]
    return cityscape, label
    
  def transform(self, image) :
    transform_ops = transforms.Compose([
      			transforms.ToTensor(),
                        transforms.Normalize(mean = (0.485, 0.56, 0.406), std = (0.229, 0.224, 0.225))
    ])
    return transform_ops(image)   