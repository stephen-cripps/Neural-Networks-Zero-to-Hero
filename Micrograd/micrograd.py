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
n = x1w1x2w2 + b; n.label = 'n'

# o = n.tanh(); o.label = 'o'
# Manually calculate tanh for clearer backProp values
e = (2*n).exp()
o = (e-1)/(e+1) 

o.backward()

# dot = draw_dot(o)
# dot.view(directory="output")

### ---- Second part - Neutons -------

x=[2.0,3.0,-1.0]
n=MLP(3,[4,4,1]) # 3 inputs into 2 layers of 4 into 1 output
print('n(x)', n(x))

draw_dot(n(x)).view(directory="output")

## Video paused at 1:51:05