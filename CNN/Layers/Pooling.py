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
        self._input_shape = input_tensor.shape
        batch, channels, h, w = input_tensor.shape
        ph, pw = self.pooling_shape
        sh, sw = self.stride
        
        out_h = (h - ph) // sh + 1
        out_w = (w - pw) // sw + 1
        
        out = np.zeros((batch, channels, out_h, out_w))
        self._max_indices = np.zeros((batch, channels, out_h, out_w, 2), dtype=int)
        
        for i in range(out_h):
            for j in range(out_w):
                hs, ws = i * sh, j * sw
                window = input_tensor[:, :, hs:hs+ph, ws:ws+pw].reshape(batch, channels, -1)
                max_idx = np.argmax(window, axis=2)
                out[:, :, i, j] = np.max(window, axis=2)
                self._max_indices[:, :, i, j, 0] = hs + max_idx // pw
                self._max_indices[:, :, i, j, 1] = ws + max_idx % pw
        
        return out

    def backward(self, error_tensor):
        grad = np.zeros(self._input_shape)
        out_h, out_w = error_tensor.shape[2:]
        batch, channels = error_tensor.shape[:2]
        
        for i in range(out_h):
            for j in range(out_w):
                h_max = self._max_indices[:, :, i, j, 0]
                w_max = self._max_indices[:, :, i, j, 1]
                grad[np.arange(batch)[:, None], np.arange(channels), h_max, w_max] += error_tensor[:, :, i, j]
        
        return grad