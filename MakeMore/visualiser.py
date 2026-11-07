import matplotlib.pyplot as plt


def visualise(N, itos):
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap='Blues')

    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, round(N[i, j].item(), 2),
                     ha="center", va="top", color="gray")

    plt.axis("off")
    plt.show()
