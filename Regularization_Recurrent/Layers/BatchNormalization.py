import numpy as np
from Layers import Base, Helpers


class BatchNormalization(Base.Base):
    def __init__(self, channels):
        """
        Batch Normalization layer.
        
        Args:
            channels: Number of channels in the input tensor
        """
        super().__init__()
        self.trainable = True
        self.channels = channels
        
        # Learnable parameters
        self.weights = None  # gamma (scale)
        self.bias = None     # beta (shift)
        
        # Gradients
        self.gradient_weights = None
        self.gradient_bias = None
        
        # Initialize parameters
        self.initialize(None, None)
        
        # Batch statistics
        self.mean = None
        self.var = None
        
        # Moving averages for testing phase
        self.moving_mean = None
        self.moving_var = None
        self.alpha = 0.8  # momentum for moving average
        
        # Cache for backward pass
        self.input_tensor = None
        self.normalized_tensor = None
        self.std = None
        
        # Store input shape for reformat
        self.input_shape = None
    
    def initialize(self, weights_initializer, bias_initializer):
        """
        Initialize weights gamma with ones and bias beta with zeros.
        
        Args:
            weights_initializer: Ignored
            bias_initializer: Ignored
        """
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
    
    def reformat(self, tensor):
        """
        Reformat between image (4D) and vector (2D) representations.
        
        Image format: (batch_size, channels, height, width)
        Vector format: (batch_size * height * width, channels)
        
        Args:
            tensor: Tensor to reformat
            
        Returns:
            Reformatted tensor
        """
        if len(tensor.shape) == 4:
            # Image to vector: (B, C, H, W) -> (B*H*W, C)
            batch_size, channels, height, width = tensor.shape
            # Transpose to (B, H, W, C)
            tensor = np.transpose(tensor, (0, 2, 3, 1))
            # Reshape to (B*H*W, C)
            tensor = tensor.reshape(-1, channels)
            return tensor
        elif len(tensor.shape) == 2:
            # Vector to image: (B*H*W, C) -> (B, C, H, W)
            # We need to recover the original shape from self.input_shape
            reshaped = tensor.reshape(self.input_shape[0], self.input_shape[2], self.input_shape[3], self.input_shape[1])
            # Transpose from (B, H, W, C) back to (B, C, H, W)
            reshaped = np.transpose(reshaped, (0, 3, 1, 2))
            return reshaped
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    
    def forward(self, input_tensor):
        """
        Forward pass through batch normalization.
        
        Args:
            input_tensor: Input tensor (batch, features) or (batch, channels, height, width)
            
        Returns:
            Normalized output tensor
        """
        self.input_shape = input_tensor.shape
        
        # Reformat if necessary (convolutional case)
        if len(input_tensor.shape) == 4:
            input_tensor = self.reformat(input_tensor)
        
        self.input_tensor = input_tensor
        
        if self.testing_phase:
            # Use moving average mean and variance
            if self.moving_mean is None:
                # First batch in test phase - should have been initialized during training
                self.moving_mean = np.zeros(self.channels)
                self.moving_var = np.ones(self.channels)
            
            # Normalize using moving averages
            normalized = (input_tensor - self.moving_mean) / np.sqrt(self.moving_var + 1e-10)
        else:
            # Training phase: use batch statistics
            # Compute batch mean and variance
            self.mean = np.mean(input_tensor, axis=0)
            self.var = np.var(input_tensor, axis=0)
            
            # Initialize moving averages on first batch
            if self.moving_mean is None:
                self.moving_mean = self.mean.copy()
                self.moving_var = self.var.copy()
            else:
                # Update moving averages
                self.moving_mean = self.alpha * self.moving_mean + (1 - self.alpha) * self.mean
                self.moving_var = self.alpha * self.moving_var + (1 - self.alpha) * self.var
            
            # Normalize using batch statistics
            eps = 1e-10
            self.std = np.sqrt(self.var + eps)
            normalized = (input_tensor - self.mean) / self.std
            self.normalized_tensor = normalized
        
        # Scale and shift
        output = self.weights * normalized + self.bias
        
        # Reformat back if necessary
        if len(self.input_shape) == 4:
            output = self.reformat(output)
        
        return output
    
    def backward(self, error_tensor):
        """
        Backward pass through batch normalization.
        
        Args:
            error_tensor: Gradient from next layer
            
        Returns:
            Gradient to pass to previous layer
        """
        # Reformat if necessary
        if len(self.input_shape) == 4:
            error_tensor = self.reformat(error_tensor)
        
        # Compute gradient w.r.t. weights and bias
        self.gradient_weights = np.sum(error_tensor * self.normalized_tensor, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)
        
        # Update weights and bias if optimizer is set
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)
        
        # Compute gradient w.r.t. input using helper function
        eps = 1e-10
        input_grad = Helpers.compute_bn_gradients(
            error_tensor, 
            self.input_tensor, 
            self.weights, 
            self.mean, 
            self.var, 
            eps=eps
        )
        
        # Reformat back if necessary
        if len(self.input_shape) == 4:
            input_grad = self.reformat(input_grad)
        
        return input_grad
