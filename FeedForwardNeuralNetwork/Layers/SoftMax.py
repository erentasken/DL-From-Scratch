import numpy as np
from .Base import Base

class SoftMax(Base):
    def __init__(self):
        pass
    
    def forward(self, input_tensor):
        # logits
        input_tensor = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        
        sum_exp = np.sum(input_tensor, axis=1, keepdims=True)
        
        self.output = input_tensor / sum_exp
        
        return self.output

    def backward(self, error_tensor):
        D = np.sum(error_tensor * self.output, axis=1, keepdims=True)
        
        E_in = self.output * (error_tensor - D)
        
        return E_in
