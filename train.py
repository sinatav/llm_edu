import torch
from minigpt import MiniGPT
from data_load import read_shakespear
import tokenizer.char_level_tokenizer as clt

data = read_shakespear()
vocab = clt.get_vocab(data)
stoi = clt.get_stoi(vocab)
itos = clt.get_itos(vocab)
vocab_size = len(stoi)

model = MiniGPT(vocab_size, d_model=64, num_heads=4, num_layers=2)

text = data[:40]
tokens = torch.tensor([clt.encode(text, stoi)])
print("Input text:", text)

logits = model(tokens)
print("Logits shape:", logits.shape)

out_ids = model.generate(tokens[:, :10], max_new_tokens=50)[0].tolist()
print("Generated:", clt.decode(out_ids, itos))
