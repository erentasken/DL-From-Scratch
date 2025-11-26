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
