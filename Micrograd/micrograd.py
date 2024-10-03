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

predictions = [mlp(x) for x in example_data]
print(predictions)
