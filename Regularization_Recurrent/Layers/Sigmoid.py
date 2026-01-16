import numpy as np
from .Base import Base


class Sigmoid(Base):
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        # Compute sigmoid and store activation (output) for backward pass
        self.activation = 1 / (1 + np.exp(-input_tensor))
        return self.activation

    def backward(self, error_tensor):
        # Gradient: sigmoid(x) * (1 - sigmoid(x)) * error_tensor = activation * (1 - activation) * error_tensor
        gradient = self.activation * (1 - self.activation) * error_tensor
        return gradient
