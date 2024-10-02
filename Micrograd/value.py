import math

class Value: 
    def __init__(self, data, _children=(), _op=(), label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label or self._build_label(_children, _op)
        self._backward = lambda: None
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self,other), '+')
        
        def backward(): 
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = backward
        return out
        
    def __mul__(self, other):
        out = Value(self.data * other.data, (self,other), '*')  
        
        def backward(): 
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) -1)/(math.exp(2*x) +1)         
        
        out = Value(t, (self,), 'tanh')
        def backward():
            self.grad += (1-t**2) * out.grad
        
        out._backward = backward
        return out
    
    def _build_label(self, parents, op):
        if len(parents) != 2:
            return ''
        
        l = list(parents)
        return f'{l[0].label} {op} {l[1].label}'
    
    def backward(self):
        sorted = []
        visited = set()
        def build_topological_list(input):
            if input not in visited:
                visited.add(input)
                for parent in input._prev:
                    build_topological_list(parent)
                    
                sorted.append(input)
        
        build_topological_list(self)

        for node in reversed(sorted):
            node._backward()
        
    
    # # My BackPropagation attempt pre-tutorial - backward was added in as part of the tutorial. Nice.
    # def back_propagate(self):
    #     if len(self._prev) == 0:
    #         return
        
    #     l = list(self._prev)
        
    #     if self._op == '*':
    #         l[0].grad = self.grad * l[1].data
    #         l[1].grad = self.grad * l[0].data
    #     elif self._op == '+':
    #         l[0].grad = self.grad
    #         l[1].grad = self.grad  
    #     elif self._op == 'tanh':
    #         l[0].grad = 1-(self.data)**2 # self is tanh of the child => 1-tanh(child.data)**2 = 1-self.data**2
            

    #     l[0].back_propagate() 
        
    #     if len(l) == 2:
    #         l[1].back_propagate()
    #         self._prev=(l[0], l[1])
    #     else:
    #         self._prev=(l[0], )
        
        
