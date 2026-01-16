import numpy as np
from Layers import Base


class Dropout(Base.Base):
    def __init__(self, probability):
        """
        Dropout layer implementing inverted dropout.
        
        Args:
            probability: Probability to keep a unit (not drop it)
        """
        super().__init__()
        self.trainable = False
        self.probability = probability
        self.mask = None
    
    def forward(self, input_tensor):
        """
        Forward pass through dropout layer.
        
        During training: randomly drop units with probability (1-p) and scale by 1/p
        During testing: pass input unchanged
        
        Args:
            input_tensor: Input activations
            
        Returns:
            Output tensor with dropout applied (training) or unchanged (testing)
        """
        if self.testing_phase:
            # During testing, pass through unchanged
            return input_tensor
        else:
            # During training: inverted dropout
            # Create mask: 1 with probability p, 0 with probability (1-p)
            self.mask = np.random.binomial(1, self.probability, input_tensor.shape)
            
            # Apply mask and scale by 1/p
            output = input_tensor * self.mask / self.probability
            return output
    
    def backward(self, error_tensor):
        """
        Backward pass through dropout layer.
        
        Apply the same mask as forward pass and scale by 1/p.
        
        Args:
            error_tensor: Gradient from next layer
            
        Returns:
            Gradient to pass to previous layer
        """
        # Apply the same mask used in forward pass and scale by 1/p
        return error_tensor * self.mask / self.probability
