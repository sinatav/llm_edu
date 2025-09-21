import torch
from minigpt import MiniGPT
from data_load import read_shakespear
import tokenizer.char_level_tokenizer as clt
import torch.nn as nn
import torch.optim as optim

data = read_shakespear()
vocab = clt.get_vocab(data)
stoi = clt.get_stoi(vocab)
itos = clt.get_itos(vocab)
vocab_size = len(stoi)

def get_batch(data, batch_size, block_size):
    inputs = []
    targets = []
    for _ in range(batch_size):
        start = torch.randint(0, len(data) - block_size - 1, (1,)).item()
        chunk = data[start:start+block_size+1]
        idxs = clt.encode(chunk, stoi)
        inputs.append(idxs[:-1])
        targets.append(idxs[1:])
    return torch.tensor(inputs, device=device), torch.tensor(targets, device=device)


d_model = 64
num_heads = 4
num_layers = 2
block_size = 128
batch_size = 32
lr = 3e-4
epochs = 100           # for demo, increase for real training
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MiniGPT(vocab_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    batch_inputs, batch_targets = get_batch(data, batch_size, block_size)

    optimizer.zero_grad()
    logits = model(batch_inputs)
    
    loss = criterion(logits.view(-1, vocab_size), batch_targets.view(-1))
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 1 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

model.eval()
prompt = "First Citizen:"
tokens = torch.tensor([clt.encode(prompt, stoi)], device=device)
out_ids = model.generate(tokens, max_new_tokens=200)[0].tolist()
print("\nGenerated text:\n", clt.decode(out_ids, itos))
