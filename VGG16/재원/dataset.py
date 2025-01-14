# 필요 라이브러리 import
import torchvision 
import torchvision.datasets as datasets #torchvision dataset 불러오기
import torchvision.transforms as transforms #이미지 변환 기능
from torch.utils.data import DataLoader #torchvision dataset 불러오기, 학습/테스트셋 준비

import matplotlib.pyplot as plt #성능 등 그래프 표시
import numpy as np 


# transforms.Compose하면 모듈화처럼 순차적으로 전처리 구현 가능
def data_transform(train = True):
    transform = transforms.Compose(
    [transforms.ToTensor(), #이미지 파일 Tensor 형태로 바꾸고 0~1 범위로 자동 정규화
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))] # RGB 평균, 표준편차 기준으로 정규화 - 입력 데이터 분포 일정하게 유지하면 학습 안정성 및 일반화 능력 향상
    )

    #torchvision 내장 CIFAR10 Dataset 활용(target_transform - 레이블은 변환 없음)
    cifar10_dataset = datasets.CIFAR10(root = "../Data/", train = train, transform=transform, target_transform=None, download = True)
    return cifar10_dataset

# 클래스 정의
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 데이터셋 이미지 시각화
def imshow(img):
    img = img / 2 + 0.5 #정규화 풀고 다시 0~1 범위로
    npimg = img.numpy() #image numpy 형태로 변형
    plt.imshow(np.transpose(npimg, (1,2,0)))#파이토치 텐서 C(채널),H,W 순서라 -> H,W,C 형태로 변형
    plt.savefig('CIFAR10_Image.png') #이미지 저장
    plt.show()
    plt.close()

## 데이터 로더에서 무작위로 이미지 가져와서 격자 형태로 시각화
def random_viualize(data_loader, batch_size):
    # 학습용 이미지를 무작위로 가져오기
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    #make_grid로 여러 이미지 grid 형태로 묶어서 출력
    imshow(torchvision.utils.make_grid(images))

    #배치 만큼의 이미지 클래스 라벨 텍스트로 변환해서 출력
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))