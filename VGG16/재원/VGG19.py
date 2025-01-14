# 실험중 E번 setting에 해당하는 VGG19 구현(Conv layer 16 + FC layer 3)
# VGG19 Architecture Summary - 3*3 Conv filter(stride = 1), MaxPooling(stride = 2), Activation(ReLU)

# 필요 라이브러리 import
import torchvision 
import torchvision.datasets as datasets #torchvision dataset 불러오기
import torchvision.transforms as transforms #이미지 변환 기능
from torch.utils.data import DataLoader #torchvision dataset 불러오기, 학습/테스트셋 준비
import matplotlib.pyplot as plt #성능 등 그래프 표시
import numpy as np 
import torch.nn as nn #파이토치 모듈 - 레이어 구성, 정규화, 손실함수, 활성화 함수 등
import torch #파이토치 library - 네트워크 학습, 자동 미분, CUDA 사용
from tqdm import trange # 모델 학습과정 tqdm 활용해서 range안에 넣고 루프 진행상황 보기


# VGG 원래 Input image 224*224 기준으로 설계, CIFAR10 image는 32*32 Size -> 이에 맞게 설계 변경

# 3*3 Conv연산 2개 layer 모듈화(Max Pooling 포함)
def conv_2_block(in_dim, out_dim): #기본 인자는 들어갈때 차원이랑, 나올 때 차원
    model = nn.Sequential( # nn.Sequential 안에 연산 순차적으로 넣어서 모듈로 묶을 수 있음
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), #3*3 kernel 활용 -> MaxPooling으로 Feature map size 줄기 때문에, 급격한 feature map size 감소 방지 위해 padding=1 추가해서 원본 사이즈 유지
        nn.ReLU(), # ReLU 활성화 함수 적용
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), # 동일하게 conv연산 한번더
        nn.ReLU(),
        nn.MaxPool2d(2,2) #2*2 size MaxPooling 적용
    )
    return model

# 3*3 Conv연산 2개 layer 모듈화(Max Pooling 포함)
def conv_4_block(in_dim, out_dim): # 위의 블록과 동일하게 기본 인자는 input 차원, output 차원
    model = nn.Sequential(
        # 이하 (Conv 연산 + 활성화함수 적용) 4번 반복 뒤 MaxPooling 까지는 동일
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), 
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

# 모델 정의
class VGG19(nn.Module):
    def __init__(self, base_dim, num_classes = 10): #CIFAR-10 데이터셋 클래스 개수 10개
        super().__init__() #상속받은 nn.Module 부모 클래스 초기화 -> 파이토치 기능 활성화
        
        #특징 추출기(Conv layer)
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim), #3개 채널 -> 64
            conv_2_block(base_dim, 2*base_dim), #64 -> 128 
            conv_4_block(2*base_dim, 4*base_dim), #128 -> 256
            conv_4_block(4*base_dim, 8*base_dim), #256 -> 512
            conv_4_block(8*base_dim, 8*base_dim) #512 -> 512
        )

        #분류기(FC layer)
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*1*1, 256), #선형 layer 512 -> 256 (기존은 4096인데, CIFAR-10 데이터셋 특성에 맞게 점진적으로 조정)
            nn.ReLU(), #활성화함수
            nn.Linear(256, 128), #256 -> 128
            nn.ReLU(), #위와 동일
            nn.Linear(128, num_classes), #최종 64 -> 10(클래스 개수만큼)
        )

    #순전파 학습 설계(기존 설계에서 기본 dropout 삭제)
    def forward(self, x):
        x = self.feature(x) #특징 추출기 거치고
        x = x.view(x.size(0), -1) #batch 크기 제외한 크기 n*n*차원 다 펼치기
        x = self.fc_layer(x) #FC layer에 전달
        return x # 최종 class당 확률 Tensor

# model 정의
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #CUDA GPU 활용
model = VGG19(base_dim=64).to(device) #위의 설계 모델(기본 차원 64) -> GPU에 올리기

# 배치 사이즈, 학습률, 에포크 지정
batch_size = 100
learning_rate = 0.00005
num_epoch = 100

# 손실함수 및 최적화함수 설정
loss_func = nn.CrossEntropyLoss() #분류 문제이기 때문에 크로스엔트로피 손실 함수 지정
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) #Adam Optimizer 활용

# transforms.Compose하면 모듈화처럼 순차적으로 전처리 구현 가능
transform = transforms.Compose(
    [transforms.ToTensor(), #이미지 파일 Tensor 형태로 바꾸고 0~1 범위로 자동 정규화
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))] # RGB 평균, 표준편차 기준으로 정규화 - 입력 데이터 분포 일정하게 유지하면 학습 안정성 및 일반화 능력 향상
)

#torchvision 내장 CIFAR10 Dataset 활용(target_transform - 레이블은 변환 없음)
cifar10_train = datasets.CIFAR10(root = "../Data/", train = True, transform=transform, target_transform=None, download = True)
cifar10_test = datasets.CIFAR10(root = "../Data/", train = False, transform=transform, target_transform=None, download = True)

# 클래스 정의
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#데이터셋 이미지 시각화
def imshow(img):
    img = img / 2 + 0.5 #정규화 풀고 다시 0~1 범위로
    npimg = img.numpy() #image numpy 형태로 변형
    plt.imshow(np.transpose(npimg, (1,2,0)))#파이토치 텐서 C(채널),H,W 순서라 -> H,W,C 형태로 변형
    plt.savefig('CIFAR10_Image.png') #이미지 저장
    plt.show()
    plt.close()

# DataLoader로 train, test set 준비, 순서 섞기
train_loader = DataLoader(cifar10_train, batch_size = batch_size, shuffle = True, num_workers = 2)
test_loader = DataLoader(cifar10_test, batch_size = batch_size, shuffle = True, num_workers = 2)

#train set 첫번째 배치 생성
dataiter = iter(train_loader)
images, labels = next(dataiter)

#make_grid로 여러 이미지 grid 형태로 묶어서 출력
imshow(torchvision.utils.make_grid(images))

#배치 만큼의 이미지 클래스 라벨 텍스트로 변환해서 출력
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

#학습
loss_arr = [] #loss 담아줄 array 생성
for i in trange(num_epoch): #100 epoch 학습
    for j,[image,label] in enumerate(train_loader): #image랑 label 불러오기

        #GPU에 이미지랑 Label 얹기
        x = image.to(device)
        y_ = label.to(device)

        optimizer.zero_grad() #이전 gradient 초기화
        output = model.forward(x) #순전파
        loss = loss_func(output, y_) #손실함수 계산
        loss.backward() #역전파
        optimizer.step() #가중치 업데이트

        # 10번째 배치마다 loss 출력 후 array에 저장
        if i % 10 == 0:
            print(loss)
            loss_arr.append(loss.cpu().detach().numpy())


# loss curve 그리기
plt.plot(loss_arr)
# loss curve 그래프 이미지 저장
plt.savefig('CIFAR10_VGG19_Loss_curve.png')
plt.show()

#정확도 계산을 위한 전체 개수, 맞은 개수 변수 생성
correct = 0
total = 0

#모델 평가 모드 전환
model.eval()

# 테스트
with torch.no_grad(): #gradient 계산 안함(학습 X)
    for image,label in test_loader:

        #테스트셋도 동일하게 Image, label 별로 GPU에 얹기
        x = image.to(device) 
        y = label.to(device)

        output = model.forward(x) #순전파 수행
        _, output_index = torch.max(output,1) #예측 확률중 가장 큰 것과 그 인덱스 반환

        total += label.size(0) #전체 배치 샘플 수
        correct += (output_index == y).sum().float() #맞은 개수 세기 

    print(total)
    print(correct)
    print("Accuracy of Test Data: {}%".format(100*correct/total)) #최종 예측 정확도 출력