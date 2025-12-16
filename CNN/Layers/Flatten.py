from .Base import Base
import numpy as np

class Flatten(Base):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape

        return input_tensor.reshape(self.input_shape[0], -1) # -1 assigns other dimensions automatically. 

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)