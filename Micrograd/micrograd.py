from value import Value
from grapher import draw_dot
    

# Inputs x1, x2
x1 =Value(2.0, label='Network Input 1')
x2 =Value(0.0, label='Network Input 2')

# Weights w1,w2
w1 =Value(-3.0, label='Weight 1')
w2 =Value(1.0, label='Weight 2')

# Neuron bias
b = Value(6.8813735870195432, label='b')

# Neuron inputs
x1w1 = x1*w1; x1w1.label='Neuron Input 1'
x2w2 = x2*w2; x2w2.label='Neuron Input 1'

# Neuron Value
x1w1_x2w2 = x1w1 + x2w2; x1w1_x2w2.label='Neuron Value'

# Weighted Neuron Value
n = x1w1_x2w2 + b; n.label='Weighted Neuron Value'

# Neuron output
#o = n.tanh(); o.label='Neuron Output'
e = ((n*2).exp())
o =  (e -1) / (e +1)

o.label='Neuron Output'
o.grad = 1.0
o.back_propagate()

draw_dot(o).render(directory='./', filename="neuron")

# Paused at 1:43:56