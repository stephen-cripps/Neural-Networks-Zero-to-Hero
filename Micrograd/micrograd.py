from multi_layer_perceptron import MLP, Neuron, Layer
from value import Value
from grapher import draw_dot  

# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
mlp = x1w1x2w2 + b; mlp.label = 'n'

# o = n.tanh(); o.label = 'o'
# Manually calculate tanh for clearer backProp values
e = (2*mlp).exp()
o = (e-1)/(e+1) 

o.backward()

# dot = draw_dot(o)
# dot.view(directory="output")

### ---- Second part - Neurons -------

xs=[
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0]
]
ys=[1.0, -1.0, -1.0, 1.0] # Desired targets

mlp=MLP(3,[4,4,1]) # 3 inputs into 2 layers of 4 into 1 output

  
for i in range(100):
  # Forward Pass (setting values)
  ypred = [mlp(x) for x in xs] 
  loss = sum((yout-ygt)**2 for ygt, yout in zip(ys, ypred))  # measure of how well the neural net is performing (sum of difference squared)
  print('loss', loss)

  # Back Pass (re-calculating  grads)
  for p in mlp.parameters():
    p.grad = 0.0
  loss.backward()
  
  # Update weights and biases based on grad
  for p in mlp.parameters():
    p.data -= 0.05 * p.grad # Update weights and biases based on grad
    
print('ypred', ypred)
