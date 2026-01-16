import numpy as np


class L1_Regularizer:
    def __init__(self, alpha):
        """
        L1 Regularizer that enforces sparsity in weights.
        
        Args:
            alpha: Regularization strength
        """
        self.alpha = alpha
    
    def calculate_gradient(self, weights):
        """
        Calculate the gradient of the L1 norm with respect to weights.
        Returns the element-wise sign of the weight tensor multiplied by alpha.
        
        Args:
            weights: Weight tensor
            
        Returns:
            Gradient of L1 norm (sign(weights) * alpha)
        """
        return np.sign(weights) * self.alpha
    
    def norm(self, weights):
        """
        Calculate the L1 norm of the weights.
        Returns sum of absolute values multiplied by alpha.
        
        Args:
            weights: Weight tensor
            
        Returns:
            L1 norm (sum(|weights|) * alpha)
        """
        return np.sum(np.abs(weights)) * self.alpha


class L2_Regularizer:
    def __init__(self, alpha):
        """
        L2 Regularizer that penalizes large weights.
        
        Args:
            alpha: Regularization strength
        """
        self.alpha = alpha
    
    def calculate_gradient(self, weights):
        """
        Calculate the gradient of the L2 norm with respect to weights.
        Returns the weight tensor multiplied by alpha.
        
        Args:
            weights: Weight tensor
            
        Returns:
            Gradient of L2 norm (weights * alpha)
        """
        return weights * self.alpha
    
    def norm(self, weights):
        """
        Calculate the L2 norm squared of the weights.
        Returns sum of squared weights multiplied by alpha.
        
        Args:
            weights: Weight tensor
            
        Returns:
            L2 norm squared (sum(weights^2) * alpha)
        """
        return np.sum(weights ** 2) * self.alpha
