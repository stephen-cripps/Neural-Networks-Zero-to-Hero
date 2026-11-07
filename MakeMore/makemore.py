import torch
from visualiser import visualise

words = open('names.txt', 'r').read().splitlines();

## Initialise empty array
N = torch.zeros((27,27), dtype = torch.int32)

## Get sorted list of all characters used
chars = sorted(list(set(''.join(words))))
## Create mapping of char to int
stoi = {s:i+1 for i,s in enumerate(chars)}

## Add special character for start/end of words
stoi['.'] = 0

## Create mapping from int to char
itos = {i: s for s, i in stoi.items()}


## Populate array of pairs of letters found in words
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip (chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        N[ix1, ix2] += 1

## Create probability array of bigrams
P = N.float()
P /= P.sum(1, keepdim=True) # Normalise rows

## Create deterministic seed for reproducability
g = torch.Generator().manual_seed(2147483647)

ix = 0

for i in range(20):
    out = []
    while True:
        ## Create normalised probability distribtuion of row ix (the previously selected letter)
        p = P[ix]

        ## Deterministically select next letter
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

        ## If end token hit
        if ix == 0: 
            break;
        
        out.append(itos[ix])

    print(''.join(out))

# visualise(N, itos)

## Video stopped at 00:50:14