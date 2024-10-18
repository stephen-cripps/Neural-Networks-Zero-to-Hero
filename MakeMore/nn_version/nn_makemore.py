from nn_version.nn_reader import read_file_and_generate_training_set
import torch.nn.functional as F
import torch

words, inputs, labels, itos, stoi, size = read_file_and_generate_training_set('names')

encoded_input = F.one_hot(inputs, num_classes=size).float()

weights = torch.randn((size, size))

print(list(itos[i.item()] for i in inputs))
print((encoded_input @ weights))

# paused at 17:37
