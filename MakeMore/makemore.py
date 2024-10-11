import torch
import matplotlib.pyplot as plt
import re

content = open('names.txt', 'r', encoding='utf-8').read().lower()
# content = open('artists.txt', 'r', encoding='utf-8').read().lower()

# Quick and dirty way to clean up chars that break the plot - may delete for the final product
cleaned_content = re.sub(r'[^a-z\s]', '?', content).replace('??', '')

words = cleaned_content.splitlines()

unique_chars = ['.']
unique_chars += sorted(list(set(''.join(words))))

size = len(unique_chars)
N = torch.zeros((size, size), dtype=torch.int32)

# A map from a char to an int
stoi = {s: i for i, s in enumerate(unique_chars)}
print(stoi)

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    idx1 = stoi[ch1]
    idx2 = stoi[ch2]
    N[idx1, idx2] += 1

itos = {i: s for s, i in stoi.items()}

# plt.figure(figsize=(16, 16))
# plt.imshow(N, cmap='Blues')
# for i in range(size):
#   for j in range(size):
#     chstr = itos[i] + itos[j]
#     plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
#     plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')

# plt.show()

# Prbability o letter appearing first
p = N[0].float()
p = p / p.sum()

print(p)

# Paused video at 26:52
