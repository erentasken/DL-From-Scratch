from .Base import Base
import numpy as np

class FullyConnected(Base):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.trainable = True

        # include bias as the last row
        self.weights = np.random.random((input_size + 1, output_size))

        self.gradient_weights = None
        self._optimizer = None

    def initialize(self, weight_initializer, bias_initializer):
        # weight matrix without bias
        fan_in = self.weights.shape[0] - 1  # last row is bias , we should remove that from dimension
        fan_out = self.weights.shape[1]

        weight_matrix = weight_initializer.initialize((fan_in, fan_out), fan_in, fan_out)

        # bias row (1 x fan_out)
        bias_vector = bias_initializer.initialize((1, fan_out), fan_in, fan_out)

        # stack together
        self.weights = np.vstack((weight_matrix, bias_vector))
    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    def forward(self, input_tensor):
        batch = input_tensor.shape[0]

        # Add bias = 1 column to input
        bias = np.ones((batch, 1))
        self.input_tensor = np.hstack((input_tensor, bias))

        # Output = X · W
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):

        # Error to pass to previous layer (ignore bias row in weights)
        error_prev = np.dot(error_tensor, self.weights[:-1].T)
        # Gradient of weights = Xᵀ · error
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)

        # Update weights if optimizer exists
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(
                self.weights, self.gradient_weights
            )

        return error_prev
