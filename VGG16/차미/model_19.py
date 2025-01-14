## VGG19로 변경

'''
VGGNet 구조

!! kernel_size=3x3, stride=1, padding=1로 고정
3x3 합성곱 연산 x2 (채널 64)
3x3 합성곱 연산 x2 (채널 128)
3x3 합성곱 연산 x (3 + 1) (채널 256)
3x3 합성곱 연산 x (3 + 1) (채널 512)
3x3 합성곱 연산 x (3 + 1) (채널 512)
FC layer x3
- FC layer 4096
- FC layer 4096
- FC layer 1000
'''

import torch
import torch.nn as nn


def conv2_block(in_dim, out_dim):
    model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1), 
            nn.ReLU(), 
            nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
    return model



def conv4_block(in_dim, out_dim):
    model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1), 
            nn.ReLU(), 
            nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1), 
            nn.ReLU(), 
            nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1), 
            nn.ReLU(), 
            nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
    return model


class VGG16(nn.Module):
    def __init__(self, base_dim, num_classes = 10):
        ## 부모 클래스인 nn.Module의 초기화 메서드 호출
        super(VGG16, self).__init__()

        ## feature: 채널 수를 증가시키면서 피처 추출
        self.feature = nn.Sequential(
            conv2_block(3, base_dim), ## base_dim = 64 / 3 -> 64
            conv2_block(base_dim, 2*base_dim), ## 64 -> 128
            conv4_block(2*base_dim, 4*base_dim), ## 128 -> 256
            conv4_block(4*base_dim, 8*base_dim), ## 256 -> 512
            conv4_block(8*base_dim, 8*base_dim) ## 512 -> 512
            )

        ## 완전 연결 레이어 정의
        self.fc_layer = nn.Sequential(
            ## CIFAR10은 이미지 크기가 32 x 32 이기 때문에
            nn.Linear(8*base_dim*1*1, 4096),
            ## 8을 곱하는 이유: 위 feature를 통과하면 2^3이 곱해지기 때문에 입력 채널 수가 8배 늘어나기 때문 
            ## 1*1인 이유: MaxPooling을 5번하면 2가 5번 나눠져 크기가 32 x 32에서 1 x 1이 되기 때문
            ## 만약 ImageNet이라면..? 
            ## -> ImageNet은 224 x 224이므로 nn.Linear(8*base_dim*7*7, 4096)

            nn.ReLU(inplace = True), 
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace = True), 
            nn.Dropout(), 
            nn.Linear(1000, num_classes)
            )

    def forward(self, x): 
        x = self.feature(x)
        ## print(x.shape)
        x = x.view(x.size(0), -1)
        ## print(x.shape)
        x = self.fc_layer(x)
        return x