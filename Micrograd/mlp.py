import random
from value import Value


class Neuron:
  def __init__(self, num_inputs) -> None:
    self.weights = [Value(random.uniform(-1, 1), label=f'weight {idx}') for idx, _ in enumerate(range(num_inputs))]
    self.bias = Value(random.uniform(-1, 1), label={'bias'})

  def __call__(self, inputs):
    activation = sum((weight * input for weight, input in zip(self.weights, inputs)), self.bias)
    return activation.tanh()

  def parameters(self):
    return self.weights + [self.bias]


class Layer:
  def __init__(self, num_inputs, num_outputs) -> None:
    self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]

  def __call__(self, inputs):
    return [n(inputs) for n in self.neurons]

  def parameters(self):
    return [param for neuron in self.neurons for param in neuron.parameters()]


class MLP:
  def __init__(self, num_inputs, num_outputs) -> None:
    sizes = [num_inputs] + num_outputs
    # Creates each layer with the num_inputs value being the output value of the previous layer.
    self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(num_outputs))]

  def __call__(self, inputs):
    for layer in self.layers:
      outputs = layer(inputs)
      inputs = outputs

    return outputs[0] if len(outputs) == 1 else outputs

  def parameters(self):
    return [param for layer in self.layers for param in layer.parameters()]
