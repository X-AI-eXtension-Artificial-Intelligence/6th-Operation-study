import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import data_transform
from model import VGG16


batch_size = 64
learning_rate = 0.0002
num_epoch = 10

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


## 데이터 로더 정의
train_loader = DataLoader(data_transform(train=True), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data_transform(train=False), batch_size=batch_size, shuffle=False)


# 모델 정의
model = VGG16(base_dim = 64).to(device)

# 손실 함수 및 옵티마이저 정의
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


## 학습
loss_arr = []
for epoch in range(num_epoch):
    epoch_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        ## 데이터 준비
        x = images.to(device)
        y_ = labels.to(device)
        
        ## 순전파
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output, y_)

        ## 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()

        ## 손실 값 저장
        epoch_loss += loss.item()

    ## 에폭별 손실
    avg_loss = epoch_loss / len(train_loader)
    loss_arr.append(avg_loss)
    print(f"Epoch [{epoch + 1}/{num_epoch}], Loss: {avg_loss:.4f}")


## 학습 손실 시각화
plt.plot(loss_arr, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss Curve")
plt.show()

## 학습 완료된 모델 저장
torch.save(model.state_dict(), "vgg16_model.pth")