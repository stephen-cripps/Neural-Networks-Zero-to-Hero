class Value: 
    def __init__(self, data, _children=(), _op=(), label=''):
        self.data = data
        self.grad=0.0
        self._prev = set(_children)
        self._op = _op
        self.label=label or self._build_label(_children, _op)
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        return Value(self.data + other.data, (self,other), '+')
        
    def __mul__(self, other):
        return Value(self.data * other.data, (self,other), '*')    
    
    def _build_label(self, children, op):
        if len(children) != 2:
            return ''
        
        l = list(children)
        return f'{l[0].label} {op} {l[1].label}'
    
    # i built this without the tutorial, I'm sure there's a much cleaner way to pick up next time
    def back_propagate(self):
        if len(self._prev) == 0:
            return
        
        l = list(self._prev)
        
        if self._op == '*':
            l[0].grad = self.grad * l[1].data
            l[1].grad = self.grad * l[0].data
        elif self._op == '+':
            l[0].grad = self.grad
            l[1].grad = self.grad  
            
        l[0].back_propagate()          
        l[1].back_propagate()
        
        self._prev={l[0], l[1]}
        
        