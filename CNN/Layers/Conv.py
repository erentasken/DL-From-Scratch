from .Base import Base
import numpy as np
import copy

class Conv(Base):
    def __init__(self, stride, kernel_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride = (stride, stride) if isinstance(stride, int) else (stride[0], stride[0]) if len(stride) == 1 else tuple(stride)
        self.kernel_shape = kernel_shape
        self.num_kernels = num_kernels
        self.is_1d = None
        self.weights = np.random.randn(num_kernels, *kernel_shape) * 0.1
        self.bias = np.random.randn(num_kernels) * 0.1
        self._input_tensor = None
        self._input_shape = None

    def initialize(self, weight_initializer, bias_initializer):
        fan_in = np.prod(self.kernel_shape)
        fan_out = np.prod(self.kernel_shape[1:]) * self.num_kernels
        self.weights = weight_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    @property
    def optimizer(self):
        return getattr(self, '_optimizer', None)

    @optimizer.setter
    def optimizer(self, opt):
        # keep original reference and create separate optimizer instances
        # for weights and bias so their internal states don't conflict
        self._optimizer = opt
        self._opt_w = copy.deepcopy(opt)
        self._opt_b = copy.deepcopy(opt)

    def _calc_pad(self, in_size, stride, k_size):
        out_size = int(np.ceil(in_size / stride))
        pad = max(0, (out_size - 1) * stride + k_size - in_size)
        return out_size, (pad // 2, pad - pad // 2)

    def _unpad(self, tensor, pad_start, pad_end, axis):
        if pad_end > 0:
            return np.take(tensor, range(pad_start, tensor.shape[axis] - pad_end), axis=axis)
        return np.take(tensor, range(pad_start, tensor.shape[axis]), axis=axis)

    def forward(self, input_tensor):
        self._input_tensor = input_tensor.copy()
        self._input_shape = input_tensor.shape
        self.is_1d = len(input_tensor.shape) == 3
        k_size = tuple(self.kernel_shape[1:])
        
        if self.is_1d:
            batch, _, in_len = input_tensor.shape
            out_len, (ps, pe) = self._calc_pad(in_len, self.stride[0], k_size[0])
            padded = np.pad(input_tensor, ((0, 0), (0, 0), (ps, pe)), mode='constant')
            out = np.zeros((batch, self.num_kernels, out_len))
            for b in range(batch):
                for k in range(self.num_kernels):
                    for l in range(out_len):
                        s = l * self.stride[0]
                        out[b, k, l] = np.sum(padded[b, :, s:s+k_size[0]] * self.weights[k]) + self.bias[k]
            return out
        else:
            batch, _, in_h, in_w = input_tensor.shape
            out_h, (phs, phe) = self._calc_pad(in_h, self.stride[0], k_size[0])
            out_w, (pws, pwe) = self._calc_pad(in_w, self.stride[1], k_size[1])
            padded = np.pad(input_tensor, ((0, 0), (0, 0), (phs, phe), (pws, pwe)), mode='constant')
            out = np.zeros((batch, self.num_kernels, out_h, out_w))
            for b in range(batch):
                for k in range(self.num_kernels):
                    for h in range(out_h):
                        for w in range(out_w):
                            hs, ws = h * self.stride[0], w * self.stride[1]
                            out[b, k, h, w] = np.sum(padded[b, :, hs:hs+k_size[0], ws:ws+k_size[1]] * self.weights[k]) + self.bias[k]
            return out

    def backward(self, error_tensor):
        k_size = tuple(self.kernel_shape[1:])
        batch = error_tensor.shape[0]
        
        if self.is_1d:
            in_len = self._input_shape[2]
            out_len, (ps, pe) = self._calc_pad(in_len, self.stride[0], k_size[0])
            padded = np.pad(self._input_tensor, ((0, 0), (0, 0), (ps, pe)), mode='constant')
            grad_pad = np.zeros_like(padded)
            self.gradient_weights = np.zeros_like(self.weights)
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))
            for b in range(batch):
                for k in range(self.num_kernels):
                    for l in range(out_len):
                        s = l * self.stride[0]
                        err = error_tensor[b, k, l]
                        patch = padded[b, :, s:s+k_size[0]]
                        self.gradient_weights[k] += err * patch
                        grad_pad[b, :, s:s+k_size[0]] += err * self.weights[k]
            grad = self._unpad(grad_pad, ps, pe, 2)
        else:
            in_h, in_w = self._input_shape[2:4]
            out_h, (phs, phe) = self._calc_pad(in_h, self.stride[0], k_size[0])
            out_w, (pws, pwe) = self._calc_pad(in_w, self.stride[1], k_size[1])
            padded = np.pad(self._input_tensor, ((0, 0), (0, 0), (phs, phe), (pws, pwe)), mode='constant')
            grad_pad = np.zeros_like(padded)
            self.gradient_weights = np.zeros_like(self.weights)
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
            for b in range(batch):
                for k in range(self.num_kernels):
                    for h in range(out_h):
                        for w in range(out_w):
                            hs, ws = h * self.stride[0], w * self.stride[1]
                            err = error_tensor[b, k, h, w]
                            patch = padded[b, :, hs:hs+k_size[0], ws:ws+k_size[1]]
                            self.gradient_weights[k] += err * patch
                            grad_pad[b, :, hs:hs+k_size[0], ws:ws+k_size[1]] += err * self.weights[k]
            grad_h = self._unpad(grad_pad, phs, phe, 2)
            grad = self._unpad(grad_h, pws, pwe, 3)
        
        if getattr(self, '_opt_w', None) is not None:
            self.weights = self._opt_w.calculate_update(self.weights, self.gradient_weights)
        elif getattr(self, '_optimizer', None) is not None:
            # fallback: use single optimizer instance
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)

        if getattr(self, '_opt_b', None) is not None:
            self.bias = self._opt_b.calculate_update(self.bias, self.gradient_bias)
        elif getattr(self, '_optimizer', None) is not None:
            self.bias = self._optimizer.calculate_update(self.bias, self.gradient_bias)
        return grad