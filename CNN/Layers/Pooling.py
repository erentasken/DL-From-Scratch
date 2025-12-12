from .Base import Base
import numpy as np

class Pooling(Base):
    def __init__(self, stride, pooling_shape):
        super().__init__()
        self.trainable = False
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.pooling_shape = (pooling_shape, pooling_shape) if isinstance(pooling_shape, int) else tuple(pooling_shape)
        self._max_indices = None
        self._input_shape = None

    def forward(self, input_tensor):
        """
        Forward pass for max-pooling layer (2D only).
        Input shape: (batch, channels, height, width)
        Output shape: (batch, channels, out_height, out_width)
        """
        self._input_shape = input_tensor.shape
        batch, channels, height, width = input_tensor.shape
        pool_h, pool_w = self.pooling_shape
        stride_h, stride_w = self.stride
        
        # Calculate output dimensions using valid padding
        out_height = (height - pool_h) // stride_h + 1
        out_width = (width - pool_w) // stride_w + 1
        
        output_tensor = np.zeros((batch, channels, out_height, out_width))
        self._max_indices = np.zeros((batch, channels, out_height, out_width, 2), dtype=int)
        
        # Perform max-pooling for each batch and channel
        for h_out in range(out_height):
            for w_out in range(out_width):
                h_start = h_out * stride_h
                w_start = w_out * stride_w
                window = input_tensor[:, :, h_start:h_start + pool_h, w_start:w_start + pool_w]
                
                # Find max values and their positions within the window
                window_flat = window.reshape(batch, channels, -1)
                max_flat_indices = np.argmax(window_flat, axis=2)
                max_vals = np.max(window_flat, axis=2)
                
                output_tensor[:, :, h_out, w_out] = max_vals
                
                # Convert flat indices to 2D coordinates
                max_h = max_flat_indices // pool_w
                max_w = max_flat_indices % pool_w
                self._max_indices[:, :, h_out, w_out, 0] = h_start + max_h
                self._max_indices[:, :, h_out, w_out, 1] = w_start + max_w
        
        return output_tensor

    def backward(self, error_tensor):
        """
        Backward pass for max-pooling layer.
        Error shape: (batch, channels, out_height, out_width)
        Returns: input_error_tensor shape (batch, channels, height, width)
        """
        batch, channels, height, width = self._input_shape
        input_error_tensor = np.zeros((batch, channels, height, width))
        
        out_height, out_width = error_tensor.shape[2:]
        
        # Route errors to the positions where max values were found
        # Use += to accumulate errors for overlapping pooling windows
        for h_out in range(out_height):
            for w_out in range(out_width):
                h_max = self._max_indices[:, :, h_out, w_out, 0]
                w_max = self._max_indices[:, :, h_out, w_out, 1]
                input_error_tensor[np.arange(batch)[:, None], np.arange(channels), h_max, w_max] += error_tensor[:, :, h_out, w_out]
        
        return input_error_tensor
