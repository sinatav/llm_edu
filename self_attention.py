import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        scores = (Q @ K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        weights = F.softmax(scores, dim=-1)
        return weights @ V
    