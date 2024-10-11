# reader.py
import torch
import re
import os

_names_cache = None
_artists_cache = None


def get_unique_chars(words):
  unique_chars = ['.']
  unique_chars += sorted(list(set(''.join(words))))
  return unique_chars


def generate_bigram_tensor(size, words, stoi):
  N = torch.zeros((size, size), dtype=torch.int32)

  for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
      idx1 = stoi[ch1]
      idx2 = stoi[ch2]
      N[idx1, idx2] += 1

  return N


def read_and_map_names():
  global _names_cache
  if _names_cache is None:
    print('No cache found - creating')
    with open('inputs/names.txt', 'r', encoding='utf-8') as file:
      words = file.read().splitlines()

    unique_chars = get_unique_chars(words)
    stoi = {s: i for i, s in enumerate(unique_chars)}
    itos = {i: s for s, i in stoi.items()}
    size = len(unique_chars)
    bigram_tensor = generate_bigram_tensor(size, words, stoi)

    _names_cache = (words, bigram_tensor, itos, stoi, size)
  return _names_cache


def read_and_map_artists():
  global _artists_cache
  if _artists_cache is None:
    print('No cache found - creating')
    with open('inputs/artists.txt', 'r', encoding='utf-8') as file:
      content = file.read().lower()
      cleaned_content = re.sub(r'[^a-z\s]', '?', content).replace('??', '')
      words = cleaned_content.splitlines()

    unique_chars = get_unique_chars(words)
    stoi = {s: i for i, s in enumerate(unique_chars)}
    itos = {i: s for s, i in stoi.items()}
    size = len(unique_chars)
    bigram_tensor = generate_bigram_tensor(size, words, stoi)

    _artists_cache = (words, bigram_tensor, itos, stoi, size)
  return _artists_cache
