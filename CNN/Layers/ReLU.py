from .Base import Base
import numpy as np

class ReLU(Base):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        relu_grad = (self.input_tensor > 0).astype(np.float32)
        return relu_grad * error_tensor