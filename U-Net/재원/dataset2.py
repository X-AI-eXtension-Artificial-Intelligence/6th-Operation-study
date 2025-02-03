## 825*550 사이즈의 Kaggle 1000장의 사람과 옷 데이터셋
## Dataset 출처 : https://www.kaggle.com/datasets/rajkumarl/people-clothing-segmentation/data


import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Data Loader 구현
class ClothingSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
  
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform  # 이미지용 transform
        self.mask_transform = mask_transform  # 마스크용 transform

        # 정렬 후 리스트 저장
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        # 데이터셋의 총 개수 반환
        return len(self.image_filenames)

    def __getitem__(self, index):
        # 파일 경로 설정
        img_path = os.path.join(self.image_dir, self.image_filenames[index])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[index])

        # 이미지 및 마스크 로드
        image = Image.open(img_path).convert("RGB")  # RGB 변환
        mask = Image.open(mask_path)  # PNG 마스크 처리

        # 마스크 변환 (RGBA → Grayscale 변환)
        if mask.mode == "RGBA":
            mask = mask.convert("L")
        elif mask.mode == "RGB":
            mask = mask.convert("L")

        # 변환 적용 (각각 다르게 적용 가능)
        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


# 텐서 변환 (마스크 전용)
class ToTensor(object):
    def __call__(self, mask):

        mask = np.array(mask)  # PIL.Image → NumPy 변환

        # NumPy에서 (H, W, C) 형태라면 (C, H, W)로 변환
        if len(mask.shape) == 3:
            mask = mask.transpose((2, 0, 1))

        mask = mask.astype(np.float32)  # dtype 변경
        return torch.from_numpy(mask)  # Tensor 변환


# 이미지 정규화 (평균 및 표준편차 적용)
class NormalizationForImage(object):
    def __init__(self, mean=0.5, std=0.5):

        self.mean = mean
        self.std = std

    def __call__(self, image):

        if not isinstance(image, torch.Tensor):  # Tensor가 아닐 경우 변환
            image = transforms.ToTensor()(image)

        return (image - self.mean) / self.std


# 마스크 정규화 (0~255 → 0~1 변환)
class NormalizationForMask(object):
    def __call__(self, mask):

        if not isinstance(mask, torch.Tensor):  # Tensor가 아닐 경우 변환
            mask = transforms.ToTensor()(mask)

        return mask / 255.0  # 마스크는 단순히 0~1 범위로 변환


