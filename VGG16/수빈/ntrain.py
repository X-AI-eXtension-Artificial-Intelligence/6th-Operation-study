import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from VGG16 import VGG16
from torchvision.datasets import ImageFolder

# train 설정 값 정의
batch_size = 50 # 배치 사이즈: 1회에 50개의 이미지 처리
learning_rate = 0.0002 # learning rate: 0.0002
num_epoch = 30 # 에포크 수: train 데이터 전체를 30회 반복 학습

# 장치 설정- CPU 또는 GPU 사용
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU 사용 가능: cuda 사용
                                                                         # GPU 사용 불가능: CPU 사용
print(device) # devcie 확인

# 이미지 데이터 전처리
transforms = transforms.Compose( 
    [transforms.Resize((224, 224)),  # 이미지 사이즈 조정
     transforms.ToTensor(),          # 이미지를 PyTorch 텐서로 변환하여, 이미지 픽셀 값을 0~1 범위로 정규화
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # 각 R, G, B 채널에 대해 
                                                        # 평균(0.5, 0.5, 0.5), 표준편차(0.5, 0.5, 0.5) 기준으로 정규화 수행
)

# 데이터셋 로드
train_data = ImageFolder(root="C:/Users/leeso/OneDrive/newdata/train/", transform = transforms)
test_data = ImageFolder(root ="C:/Users/leeso/OneDrive/newdata/test/", transform = transforms)

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
# cifar10 학습 데이터셋을 배치 단위로 불러옴. 
# shuffle = True 해서 데이터 순서 랜덤하게 변경
test_loader = DataLoader(test_data, batch_size = batch_size)
# cifar10 평가 데이터셋을 배치 단위로 불러옴. 
# 평가 데이터는 shuffle X
# 클래스 수 확인 및 정의
num_classes = len(train_data.classes)  # train_data에서 클래스 수 계산
print("Number of Classes:", num_classes)

# 모델/손실 함수/옵티마이저 정의
model = VGG16(base_dim = 64, num_classes=num_classes).to(device)
# VGG16 모델을 학습에 사용
# base dim = 64: 첫 번째 합성곱 블록의 출력 채널 수
loss_func = nn.CrossEntropyLoss() # loss fuction: CrossEntropy 사용-분류 문제에 적합
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) # optimizer: Adam 옵티마이즈 사용-빠른 학습 속도, 안정적인 최적화 알고리즘



# 학습 루프
loss_arr = [] # loss array-손실 값 저장용 빈 리스트

for i in range(num_epoch):                            # 앞서 지정한 epoch만큼 반복
    for j, [image, label] in enumerate(train_loader): # 배치 단위로 데이터 가져옴
        x = image.to(device)                          # 입력 이미지 device로 이동
        y = label.to(device)                          # 입력 레이블 device로 이동

        optimizer.zero_grad()        # 이전 배치의 gradient 초기화
        output = model.forward(x)    # 모델 순전파 수행하여, 입력 데이터 모델에 전달
        loss = loss_func(output, y)  # 예측 값과 실제 값 간의 loss 차이를 계산
        loss.backward()              # 손실 값 기반으로 기울기 계산(역전파)
        optimizer.step()             # 계산된 기울기를 사용하여 모델의 가중치 업데이트
        

    if i % 10 == 0:                                  # epoch 10번마다, 중간 손실 출력
        print(f'epoch {i} loss :', loss)                
        loss_arr.append(loss.detach().cpu().numpy()) # loss 값 저장
        

# 모델 저장
torch.save(model.state_dict(), "./VGG16_100.pth")