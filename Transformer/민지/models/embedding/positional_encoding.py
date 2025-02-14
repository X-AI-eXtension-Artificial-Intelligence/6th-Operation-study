import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        # 인풋이랑 똑같은 크기로 pos encoding 매트릭스 생성
        self.encoding = torch.zeros(max_len, d_model, device=device) # Tx512
        # 학습시 반영 안되도록
        self.encoding.requires_grad = False
        # 0~max_len-1까지의 정수 생성
        pos = torch.arange(0, max_len, device=device)
        # 위의 정수를 실수형 텐서로 변환하고, 차원 하나 더 추가 => 2차원 텐서로 변경 (T,) => (T,1)
        pos = pos.float().unsqueeze(dim=1)

        # 짝수 인덱스만 가져옴 -> sin
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, length = x.size()

        return self.encoding[:length, :]
