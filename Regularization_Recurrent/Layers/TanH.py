import numpy as np
from .Base import Base


class TanH(Base):
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        # Compute tanh and store activation (output) for backward pass
        self.activation = np.tanh(input_tensor)
        return self.activation

    def backward(self, error_tensor):
        # Gradient: (1 - tanh^2(x)) * error_tensor = (1 - activation^2) * error_tensor
        gradient = (1 - self.activation ** 2) * error_tensor
        return gradient
