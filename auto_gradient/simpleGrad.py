import random
import numpy as np


class Val:

    def __init__(self ,data, children = () , opr = None , label = None, grad = 0.0):
        self.data = data
        self.opr = opr
        self.grad = grad
        self.label = label
        self._backward = lambda : None
        self._prev = set(children)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Val(other)
        res = Val(self.data + other.data , (self, other) , opr = '+')
        def _backward():
            self.grad += 1.0 * res.grad
            other.grad += 1.0 * res.grad
        res._backward = _backward
        return res
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Val(other)
        res = Val(self.data * other.data , (self, other) , opr = '*')
        def _backward():
            self.grad += other.data * res.grad
            other.grad += self.data * res.grad
        res._backward = _backward
        return res
    
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            other = Val(other)
        res = Val(self.data ** other.data , (self, other) , opr = f'**{other.data}')
        def _backward():
            self.grad += (other.data * (self.data ** (other.data - 1))) * res.grad
            # other.grad += res.data * np.log(self.data) * res.grad
        res._backward = _backward
        return res
    
    def __neg__(self):
        return Val(-self.data , (self,) , opr = '-')
    
    def __sub__(self, other):
        # return self + (-1 * other)
        return self + (- other)
    
    def __truediv__(self, other):
        # return Val(self.data / other.data , (self, other) , opr = '/')
        return self * other ** Val(-1)
    
    def __repr__(self):
        if self.label and self.opr:
            return f'Val(d={self.data.__repr__()}, opr={self.opr.__repr__()}, label={self.label.__repr__()})'
        if self.label:
            return f'Val(d={self.data.__repr__()}, label={self.label.__repr__()})'
        if self.opr:
            return f'Val(d={self.data.__repr__()}, opr={self.opr.__repr__()})'
        return f'Val(d={self.data.__repr__()})'
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __rtruediv__(self, other):
        return Val(other) / self
    
    def __rpow__(self, other):
        return Val(other) ** self
    

    def exp(self):
        res = Val(np.exp(self.data) , (self,) , opr = 'exp')
        def _backward():
            self.grad += res.data * res.grad
        res._backward = _backward
        return res

    def tanh(self):
        res = Val(np.tanh(self.data) , (self,) , opr = 'tanh')
        def _backward():
            self.grad += (1 - res.data ** 2) * res.grad
        res._backward = _backward
        return res
    
    def sigmoid(self):
        res = Val(1 / (1 + np.exp(-self.data)) , (self,) , opr = 'sigm')
        def _backward():
            self.grad += (res.data * (1 - res.data)) * res.grad
        res._backward = _backward
        return res
    
    def relu(self):
        res = Val(np.maximum(0, self.data) , (self,) , opr = 'relu')
        def _backward():
            self.grad += (self.data > 0) * res.grad
        res._backward = _backward
        return res
    
    def backward(self):
        topo = []
        visited = set()
        def dfs(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    dfs(child)
                topo.append(v)
        dfs(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self , n_inputs):
        self.w = [ Val(random.uniform(-1, 1)) for i in range(n_inputs)] 
        self.b = Val(random.uniform(-1, 1))
    
    def forward(self, xs):
        # res = np.dot(inputs, self.w) + self.b
        res = sum([x * w for x, w in zip(xs, self.w)], self.b)
        res2 = res.tanh()
        return res2
    
    def __repr__(self):
        return f'Neuron(w = {self.w.__repr__()} , b = {self.b.__repr__()})'
    
    def __call__(self, inputs):
        return self.forward(inputs)
    
    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self , n_inputs , n_output):
        self.neurons = [Neuron(n_inputs) for _ in range(n_output)]
    
    def forward(self, inputs):
        res = [neuron(inputs) for neuron in self.neurons]
        if len(res) == 1:
            return res[0]
        return res

    def __call__(self, inputs):
        return self.forward(inputs)

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    def __repr__(self):
        return f'Layer({self.neurons.__repr__()})'
    
class MultiLayerPerceptron:
    def __init__(self , n_inputs ,n_outputs):
        sz = [n_inputs] + n_outputs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz) - 1)]
        
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
    
    def __call__(self, inputs):
        return self.forward(inputs)
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f'MLP({self.layers.__repr__()})'

if __name__ == '__main__':
    a = Val(1, label = 'a')
    b = Val(2.5 , label = 'b')
    c = Val(3.14 , label = 'c')
    c
    d = a + b ; d.label = 'd'
    f'{d = }'
    e = b + c ; e.label = 'e'
    f'{e = }'
    f = d * e ; f.label = 'f'
    f'{f = }'
    g = f ** Val(2)
    f'{g = }' ; g.label = 'g'
    g.backward()