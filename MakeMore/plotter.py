import matplotlib.pyplot as plt


def plot_bigrams(bigrams, size, itos):
  plt.figure(figsize=(16, 16))
  plt.imshow(bigrams, cmap='Blues')
  for i in range(size):
    for j in range(size):
      chstr = itos[i] + itos[j]
      plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
      plt.text(j, i, bigrams[i, j].item(), ha="center", va="top", color='gray')

  plt.show()
