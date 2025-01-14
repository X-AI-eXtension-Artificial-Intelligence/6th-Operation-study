import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np


## 데이터 전처리를 위한 Transform 정의
def data_transform(train = True): 
    transform = transforms.Compose(
                ## 이미지를 Pytorch 텐서로 변환하고 픽셀 값을 [0,1] 범위로 정규화
                [transforms.ToTensor(), 
                ## 평균값 (0.5, 0.5, 0.5)과 표준편차 (0.5, 0.5, 0.5)를 사용해 RGB 각 채널 정규화 ([-1, 1] 사이)
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                )

    # CIFAR10 데이터 정의
    cifar10_dataset = datasets.CIFAR10(root="./data/", train=train, transform=transform, target_transform=None, download=True)
    return cifar10_dataset


## CIFAR-10 클래스
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


## 이미지 보여주기 위한 함수
def imshow(img): 
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    ## 채널 순서를 (채널, 높이, 너비) 에서 (높이, 너비, 채널)로 변경
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    plt.show()


## 데이터 로더에서 무작위로 이미지와 라벨을 가져오고 시각화
def random_viualize(data_loader, batch_size):
    # 학습용 이미지를 무작위로 가져오기
    dataiter = iter(data_loader)
    images, labels = dataiter.next()

    # 이미지 보여주기
    imshow(torchvision.utils.make_grid(images))

    # 정답(label) 출력
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))