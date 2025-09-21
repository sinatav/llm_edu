import numpy as np
from collections import defaultdict
import tokenizer.char_level_tokenizer as clt
from data_load import read_shakespear

def build_bigram_counts(data):
    counts = defaultdict(lambda: defaultdict(int))
    for ch1, ch2 in zip(data, data[1:]):
        counts[ch1][ch2] += 1
    return counts

def get_probs(char, counts):
    dist = counts[char]
    total = sum(dist.values())
    probs = {c: v/total for c, v in dist.items()}
    return probs

def predict_next(probs):
    max_value = max(probs.values())
    return [(k, v) for k, v in probs.items() if v == max_value]



# data = "je m\'appelle Sina et je suis en train de travailler"
data = read_shakespear()
vocab = clt.get_vocab(data)
stoi = clt.get_stoi(vocab)
itos = clt.get_itos(vocab)
encoded = clt.encode(data, stoi)
counts = build_bigram_counts(encoded)
probs = get_probs(stoi['a'], counts)
print(predict_next(probs))
