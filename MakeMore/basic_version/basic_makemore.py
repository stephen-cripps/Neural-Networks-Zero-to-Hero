import torch
from basic_version.basic_reader import read_and_map_names, read_and_map_artists
from plotter import plot_bigrams

words, N, itos, stoi, size = read_and_map_artists()

# plot_bigrams(N, size, itos)

g = torch.Generator().manual_seed(2147483647)

probability_matrix = (N + 1).float()
# Probability_matrix.sum(1, keepdim=True) gets the sum of each row and prduces a 27x1 vector.
# Then there's some matrix arithmetic to divide the 2
probability_matrix /= probability_matrix.sum(1, keepdim=True)

for i in range(10):
  ix = 0
  out = []
  while True:
    p = probability_matrix[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))

log_likelyhood = 0.0
n = 0

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    idx1 = stoi[ch1]
    idx2 = stoi[ch2]
    prob = probability_matrix[idx1, idx2]
    log_likelyhood += torch.log(prob)
    n += 1

print(-log_likelyhood / n)
