import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from VGG16 import VGG16


# 장치 설정- CPU 또는 GPU 사용
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu') # GPU 사용 가능: cuda 사용
                                                                        # GPU 사용 불가능: CPU 사용
print(device)

# 배치 사이즈 설정
batch_size = 100 # 1회에 100개의 이미지 처리

# 이미지 데이터 전처리
transform = transforms.Compose(
    [transforms.ToTensor(), # 이미지를 PyTorch 텐서로 변환하여, 이미지 픽셀 값을 0~1 범위로 정규화
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))] # 각 R, G, B 채널에 대해 
                                                        # 평균(0.5, 0.5, 0.5), 표준편차(0.5, 0.5, 0.5) 기준으로 정규화 수행
)

# 데이터셋 로드
cifar10_test = datasets.CIFAR10(root = "./Data/", train = False, transform = transforms, target_transform = None, download = True)

test_loader = DataLoader(cifar10_test, batch_size = batch_size)


# Train
# 모델 로드
model = VGG16(base_dim = 64).to(device)
# VGG16 모델을 학습에 사용
# base dim = 64: 첫 번째 합성곱 블록의 출력 채널 수

# 저장된 가중치 불러오기
model.load_state_dict(torch.load('./VGG16_100.pth'))

# 변수 초기화
correct = 0   # 정답 라벨 수 담는 변수
total = 0     # 전체 라벨 사이즈 담는 변수


model.eval() # 모델을 평가 모드로 전환
             # 평가 모드에서는 Dropout, BatchNormalization 등이 비활성화됨


# 모델 추론
with torch.no_grad(): # 역전파를 계산하지 않아서 메모리를 줄여서 절약 & 평가 속도를 향상시킴
    for i, [image, label] in enumerate(test_loader): # 배치 단위로 데이터를 가져옴
        x = image.to(device)                         # x = 이미지를 device로 이동
        y = label.to(device)                         # y = 라벨을 device로 이동

        output = model.forward(x)                    # 모델을 통과한 x의 output
        _, output_index = torch.max(output, 1)       # 모델의 출력값 중 가장 높은 확률값을 가진 클래스의 인덱스 반환
        total += label.size(0)                       # 총 데이터 수 업데이트
        correct += (output_index ==y).sum().float()  # 정답 개수 업데이트

 # 정확도 출력
    print("Accuracy of Test Data : {}%".format(100*correct/total))