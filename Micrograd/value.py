import math


class Value:
  def __init__(self, data, _children=(), _op=(), label=''):
    self.data = data
    self.grad = 0.0
    self._prev = set(_children)
    self._op = _op
    self.label = label
    self._backward = lambda: None

  def __repr__(self):
    return f"Value(data={self.data})"

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def backward():
      self.grad += out.grad
      other.grad += out.grad

    out._backward = backward
    return out

  def __radd__(self, other):
    return self + other

  def __sub__(self, other):
    return self + (-other)

  def _neg__(self):
    return self * -1

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad

    out._backward = backward
    return out

  def __rmul__(self, other):
    return self * other

  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')

    def backward():
      self.grad += out.data * out.grad

    out._backward = backward
    return out

  def tanh(self):
    x = self.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

    out = Value(t, (self,), 'tanh')

    def backward():
      self.grad += (1 - t**2) * out.grad

    out._backward = backward
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "Only supporting ints and floats"
    out = Value(self.data**other, (self,), f'**{other}')

    def backward():
      self.grad += other * self.data**(other - 1) * out.grad

    out._backward = backward
    return out

  def __truediv__(self, other):
    return self * other**-1

  def back_propagate(self):
    self.grad = 1
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
