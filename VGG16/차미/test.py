import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import data_transform
from model import VGG16

## 하이퍼파라미터 설정
batch_size = 32

## 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## 데이터 로더 정의
test_loader = DataLoader(data_transform(train=False), batch_size=batch_size, shuffle=False)

## 모델 정의 및 로드
model = VGG16(base_dim=64).to(device)
model.load_state_dict(torch.load("vgg16_model.pth"))  ## 학습된 가중치 불러오기
model.eval()  ## 평가 모드로 전환

## 손실 함수 정의
loss_func = nn.CrossEntropyLoss()

## 테스트 과정
correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():  ## 그래디언트 계산 비활성화 -> 메모리 사용량 줄이고 속도 높임
    for images, labels in test_loader:
        ## 데이터 준비
        x = images.to(device)
        y_ = labels.to(device)

        ## 모델 출력
        outputs = model(x)
        loss = loss_func(outputs, y_)
        test_loss += loss.item()

        ## 예측 및 정확도 계산
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == y_).sum().item()

## 평균 손실 및 정확도 출력
avg_loss = test_loss / len(test_loader)
accuracy = 100 * correct / total

print(f"Test Loss: {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")