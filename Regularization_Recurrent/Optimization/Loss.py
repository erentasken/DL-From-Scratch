import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.input_tensor = None
    
    def forward(self, input_tensor, label_tensor):
        """
        Compute cross entropy loss.
        
        Args:
            input_tensor: Predicted probabilities (batch_size, num_classes)
            label_tensor: One-hot encoded labels (batch_size, num_classes)
            
        Returns:
            Cross entropy loss value
        """
        self.input_tensor = input_tensor
        # Clip to avoid log(0)
        epsilon = 1e-15
        input_clipped = np.clip(input_tensor, epsilon, 1 - epsilon)
        # Cross entropy: -sum(label * log(pred))
        loss = -np.sum(label_tensor * np.log(input_clipped))
        return loss
    
    def backward(self, label_tensor):
        """
        Compute gradient of cross entropy loss.
        
        Args:
            label_tensor: One-hot encoded labels (batch_size, num_classes)
            
        Returns:
            Gradient with respect to input
        """
        # Gradient: -label / input
        epsilon = 1e-15
        input_clipped = np.clip(self.input_tensor, epsilon, 1 - epsilon)
        return -label_tensor / input_clipped
