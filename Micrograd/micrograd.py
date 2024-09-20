from value import Value
from grapher import draw_dot
        
a = Value(2, label='a')
b = Value(-3, label='b')
c = Value(10, label='c')
e = a*b; e.label = 'e'
d = e+c; d.label ='d'
f = Value(-2, label='f')
L = f*d; L.label = 'L'
L.grad = 1.0
L.back_propagate()

draw_dot(L).render(directory='./')

# Tutorial stopped at 49:27
