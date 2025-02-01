import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from dataset2 import *
from UNET2 import UNET2
from util import *
from torchvision import transforms
import torch.optim as optim

# 데이터 경로 설정
base_dir = "/home/work/XAI_WinterStudy/U-Net"
image_dir = "/home/work/XAI_WinterStudy/U-Net/png_images/IMAGES"
mask_dir = "/home/work/XAI_WinterStudy/U-Net/png_masks/MASKS"
result_dir = os.path.join(base_dir, "test_results")
os.makedirs(result_dir, exist_ok=True)

# 이미지 및 마스크 변환 (훈련 코드와 동일하게 적용해야 함)
image_transform = transforms.Compose([
    transforms.Resize((512, 256)),
    NormalizationForImage(mean=0.5, std=0.5)
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 256)),
    transforms.ToTensor(),
    NormalizationForMask()
])

# 데이터셋 로드
dataset = ClothingSegmentationDataset(image_dir, mask_dir, image_transform=image_transform, mask_transform=mask_transform)

# 저장된 테스트셋 인덱스 불러오기
test_indices_path = os.path.join(base_dir, "test_indices.json")
with open(test_indices_path, "r") as f:
    test_indices = json.load(f)

# 저장된 인덱스로 테스트셋 Subset 생성
xtest = Subset(dataset, test_indices)

# DataLoader 생성
test_loader = DataLoader(xtest, batch_size=8, shuffle=False)

# 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = UNET2(3, 59).to(device)

# 손실함수 정의하기(크로스엔트로피)
fn_loss = nn.CrossEntropyLoss().to(device)

# Optimizer 설정하기(SGD)
optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)

# 저장된 모델 가중치 불러오기
ckpt_dir = os.path.join(base_dir, "checkpoint")
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optimizer)

# IoU 계산 함수
def calculate_iou(pred_mask, true_mask, num_classes=59):
    iou_per_class = []
    
    for c in range(num_classes):
        pred_c = (pred_mask == c).astype(np.uint8)
        true_c = (true_mask == c).astype(np.uint8)
        
        intersection = np.logical_and(pred_c, true_c).sum()
        union = np.logical_or(pred_c, true_c).sum()
        
        if union == 0:
            iou = 1 if intersection == 0 else 0
        else:
            iou = intersection / union
        
        iou_per_class.append(iou)
    
    return np.nanmean(iou_per_class)

# 테스트 실행
net.eval()
iou_list = []
thresholds = np.arange(0.5, 1.0, 0.05)
precision_list = {t: [] for t in thresholds}

with torch.no_grad():
    for idx, (images, masks) in enumerate(test_loader):
        images, masks = images.to(device), masks.to(device)

        outputs = net(images)
        outputs = torch.softmax(outputs, dim=1)  # 확률 변환
        preds = torch.argmax(outputs, dim=1).cpu().numpy()  # 예측 마스크
        
        masks = masks.squeeze(1).cpu().numpy()

        for i in range(images.shape[0]):
            pred_mask = preds[i]
            true_mask = masks[i]

            # IoU 계산
            iou = calculate_iou(pred_mask, true_mask)
            iou_list.append(iou)

            # 다양한 임계값에서 precision 계산
            for t in thresholds:
                precision_list[t].append(1 if iou >= t else 0)

            # 이미지 저장
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(images[i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
            ax[0].set_title("Input Image")
            ax[1].imshow(true_mask, cmap="tab20")
            ax[1].set_title("Ground Truth")
            ax[2].imshow(pred_mask, cmap="tab20")
            ax[2].set_title(f"Prediction (IoU: {iou:.2f})")

            plt.savefig(os.path.join(result_dir, f"result_{idx}_{i}.png"))
            plt.close()

# mAP 계산
mean_precision = {t: np.mean(precision_list[t]) for t in thresholds}
map_value = np.mean(list(mean_precision.values()))

print(f"mAP Score: {map_value:.4f}")
