import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import get_loader  # 데이터 로더 함수 사용
from model import Unet  # 모델
import os
import joblib
from sklearn.cluster import KMeans
kmeans = joblib.load('../XAI/Unet/kmeans_model.pkl')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    model_name = '../XAI/Unet/unet_model.pth'
    model = Unet().to(device)  # num_classes 제거
    model.load_state_dict(torch.load(model_name))

    # 데이터셋 및 데이터 로더 설정
    val_image_dir = '../XAI/Unet/dataset/Flood/val/Image'  # 검증 이미지 데이터셋 경로
    val_mask_dir = '../XAI/Unet/dataset/Flood/val/Mask'    # 검증 마스크 데이터셋 경로
    test_batch_size = 4
    data_loader = get_loader(val_image_dir, val_mask_dir, kmeans, batch_size=test_batch_size)

    X, Y = next(iter(data_loader))
    X, Y = X.to(device), Y.to(device)
    Y_pred = model(X)
    Y_pred = torch.argmax(Y_pred, dim=1)

    # 이미지의 변환을 역으로 적용
    inverse_transform = transforms.Compose([
        transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
    ])

    fig, axes = plt.subplots(test_batch_size, 3, figsize=(15, 5 * test_batch_size))
    iou_scores = []

    for i in range(test_batch_size):
        landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
        label_class = Y[i].cpu().detach().numpy()
        label_class_predicted = Y_pred[i].cpu().detach().numpy()

        # IOU 점수 계산
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
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
        axes[i, 2].axis('off')

    plt.show()

    print(f'Average IoU Score: {sum(iou_scores) / len(iou_scores):.4f}')

if __name__ == '__main__':
    main()
