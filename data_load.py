import requests
import re

def read_shakespear():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
    response = requests.get(url)
    return response.text

def get_words(text, unique=True):
    if not unique:
        return re.findall(r'\b\w+\b', text.lower())
    return list(set(re.findall(r'\b\w+\b', text.lower())))
