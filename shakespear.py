import requests
import re
import char_level_tokenizer as clt


url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
response = requests.get(url)
text = response.text

# words = re.findall(r'\b\w+\b', text.lower())
chars = clt.get_vocab(text)

stoi = clt.get_stoi(chars)
itos = clt.get_itos(chars)

print(clt.encode("Bullocks!", stoi))
print(clt.decode([37, 53, 59, 1, 45, 53, 58, 1, 30, 47, 41, 49, 56, 53, 50, 50, 43, 42], itos))
