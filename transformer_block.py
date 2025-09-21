import torch
import torch.nn as nn
import torch.nn.functional as F
from data_load import read_shakespear
import tokenizer.char_level_tokenizer as clt

data = read_shakespear()
vocab = clt.get_vocab(data)
stoi = clt.get_stoi(vocab)
itos = clt.get_itos(vocab)

vocab_size = len(stoi)


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=5000):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        return self.token_emb(x) + self.pos_emb(positions)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        H, d_k = self.num_heads, self.d_k

        Q = self.Wq(x).view(B, T, H, d_k).transpose(1, 2)
        K = self.Wk(x).view(B, T, H, d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, H, d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ V

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(out), weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=128):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, attn_weights = self.attn(x)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x, attn_weights


def main():
    d_model = 32
    num_heads = 4
    embed = TokenAndPositionEmbedding(vocab_size, d_model)
    block = TransformerBlock(d_model, num_heads)

    text = data[:20]
    tokens = torch.tensor([clt.encode(text, stoi)])
    print("Input text:", text)

    x = embed(tokens)
    out, attn_weights = block(x)

    print("Output shape:", out.shape)
    print("Attention weights shape:", attn_weights.shape)

if __name__ == "__main__":
    main()