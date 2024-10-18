# reader.py
import torch

_cache = {}


def get_unique_chars(words):
  unique_chars = ['.']
  unique_chars += sorted(list(set(''.join(words))))
  return unique_chars


def generate_training_set(words, stoi):
  inputs, labels = [], []

  for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
      idx1 = stoi[ch1]
      idx2 = stoi[ch2]
      inputs.append(idx1)
      labels.append(idx2)

  return inputs, labels


def read_file_and_generate_training_set(filename):
  global _cache
  if filename not in _cache:
    print('No cache found - creating')
    with open(f'inputs/{filename}.txt', 'r', encoding='utf-8') as file:
      words = file.read().splitlines()

    unique_chars = get_unique_chars(words)
    stoi = {s: i for i, s in enumerate(unique_chars)}
    itos = {i: s for s, i in stoi.items()}
    size = len(unique_chars)
    inputs, labels = generate_training_set(words, stoi)

    _cache[filename] = (words, torch.tensor(inputs), torch.tensor(labels), itos, stoi, size)
  return _cache[filename]
