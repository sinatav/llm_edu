def get_vocab(text):
    return sorted(set(text))

def get_stoi(vocab):
    return {ch: i for i, ch in enumerate(vocab)}

def get_itos(vocab):
    stoi = get_stoi(vocab)
    return {i: ch for ch, i in stoi.items()}

def encode(s, stoi): return [stoi[c] for c in s]
def decode(ids, itos): return "".join(itos[i] for i in ids)
