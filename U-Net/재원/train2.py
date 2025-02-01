import os
import numpy as np


import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split #이미지, mask 데이터셋 train, test 분리되어 있지 않음


# 만들어둔 패키지 import
from UNET2 import UNET2
from dataset2 import *
from util import *

import matplotlib.pyplot as plt 
from torchvision import transforms, datasets

# 훈련 파라미터 설정
lr = 1e-5
batch_size = 8
num_epoch = 30

base_dir = '/home/work/XAI_WinterStudy/U-Net'
image_dir = '/home/work/XAI_WinterStudy/U-Net/png_images/IMAGES'
mask_dir = '/home/work/XAI_WinterStudy/U-Net/png_masks/MASKS'
ckpt_dir = os.path.join(base_dir, "checkpoint")

os.makedirs(ckpt_dir, exist_ok=True)

# 훈련을 위한 전처리와 DataLoader 구성(이미지,마스크 따로)
image_transform = transforms.Compose([
    transforms.Resize((512, 256)),  # 크기 조정
    NormalizationForImage(mean=0.5, std=0.5)  # 이미지 정규화 적용
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 256)),  # 크기 조정
    transforms.ToTensor(),  # 텐서 변환
    NormalizationForMask()  # 마스크 정규화 적용
])


dataset = ClothingSegmentationDataset(image_dir, mask_dir, image_transform = image_transform,
                                    mask_transform = mask_transform)

# 8:1:1로 train,val,split 랜덤 분할
test_size = int(0.1 * len(dataset))
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size - test_size
xtrain, xval, xtest = random_split(dataset, [train_size, val_size, test_size])

# xtest의 인덱스 저장
test_indices = xtest.indices  # random_split은 Subset을 반환하며, indices 속성을 갖고 있음
with open(os.path.join(base_dir, "test_indices.json"), "w") as f:
    json.dump(test_indices, f)

train_loader = DataLoader(xtrain, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(xval, batch_size=batch_size, shuffle=False)

# 네트워크 생성하기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNET2(3,59).to(device)

# 손실함수 정의하기(크로스엔트로피)
fn_loss = nn.CrossEntropyLoss().to(device)

# Optimizer 설정하기(SGD)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)



# 네트워크 학습(에포크 0)
st_epoch = 0

# 만약 학습한 모델이 있다면 모델 로드
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optimizer) 

for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train()  # 훈련 모드
    epoch_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        masks = masks.squeeze(1)
        masks = masks.long()

        # Forward pass
        outputs = net(images)

        # Compute loss
        loss = fn_loss(outputs, masks)

        # 계산된 손실 추가
        epoch_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 평균 손실 출력 (← for 루프 내부에서 벗어나도록 들여쓰기 수정)
    print(f"Epoch: {epoch}/{num_epoch}, Loss: {epoch_loss/len(train_loader):.4f}")

    # 검증 과정 (매 5 Epoch마다 실행)
    if epoch % 5 == 0:
        with torch.no_grad():
            net.eval()
            val_loss = 0

            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                masks = masks.squeeze(1)
                masks = masks.long()
                outputs = net(images)
                loss = fn_loss(outputs, masks)

                # 계산된 손실 추가
                val_loss += loss.item()

        # 검증 손실 출력
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

    # 5 Epoch마다 모델 저장
    if epoch % 5 == 0:
        save(ckpt_dir=ckpt_dir, net=net, optim=optimizer, epoch=epoch)
