import copy
import torch.nn as nn

## n 개의 인코더 블록으로 구성된 인코더
class Encoder(nn.Module):
    
    def __init__(self, encoder_block, n_layer, norm):
        super(Encoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(self.n_layer)])
        self.norm = norm

    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        out = self.norm(out)
        return out