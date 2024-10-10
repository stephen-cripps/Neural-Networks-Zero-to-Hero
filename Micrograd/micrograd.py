from mlp import MLP
from grapher import draw_dot

mlp = MLP(3, [4, 4, 1])

example_data = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]

desired_outputs = [1.0, -1.0, -1.0, 1.0]

for _ in range(100):
  outputs = [mlp(x) for x in example_data]
  loss = sum((output - desired_output)**2 for desired_output, output in zip(desired_outputs, outputs))
  loss.back_propagate()
  for p in mlp.parameters():
    p.data += -0.05 * p.grad
    p.grad = 0.0

print(outputs)
