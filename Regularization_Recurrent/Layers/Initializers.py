import numpy as np

class Constant:
    def __init__(self, value=0.1):
        self.value = value

    def initialize(self, shape, fan_in=None, fan_out=None): # that'll for bias init.
        return np.full(shape, self.value)

class UniformRandom:
    def initialize(self, shape, fan_in=None, fan_out=None):
        return np.random.rand(*shape)  # values in [0,1)

class Xavier: # sigmoid and tanh
    def initialize(self, shape, fan_in, fan_out):
        std = np.sqrt(2 / (fan_in + fan_out))
        return np.random.randn(*shape) * std

class He: # ReLU , forward propagation , 
    def initialize(self, shape, fan_in, fan_out=None):
        std = np.sqrt(2) / np.sqrt(fan_in)
        return np.random.randn(*shape) * std