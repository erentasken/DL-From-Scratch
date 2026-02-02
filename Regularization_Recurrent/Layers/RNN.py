import numpy as np
from .Base import Base
from .FullyConnected import FullyConnected

class RNN(Base):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size

        # Hidden state
        self.hidden_state = np.zeros((1, hidden_size))
        self._memorize = False

        # Optimizer
        self._optimizer = None

        # Fully connected layers
        self.fc_hidden = FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_output = FullyConnected(hidden_size, output_size)

        # Store intermediate states for BPTT
        self.input_tensor = None
        self.hidden_states, self.hidden_activations = [], []
        self.fc_hidden_inputs, self.fc_output_inputs = [], []

    @property
    def memorize(self): return self._memorize
    @memorize.setter
    def memorize(self, val): self._memorize = bool(val)

    @property
    def optimizer(self): return self._optimizer
    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
        self.fc_hidden.optimizer = None
        self.fc_output.optimizer = None

    @property
    def weights(self):
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, value):
        self.fc_hidden.weights = value

    @property
    def gradient_weights(self):
        return self.fc_hidden.gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self.fc_hidden.gradient_weights = value

    def initialize(self, weight_initializer, bias_initializer):
        self.fc_hidden.initialize(weight_initializer, bias_initializer)
        self.fc_output.initialize(weight_initializer, bias_initializer)

    def forward(self, input_tensor):
        time_steps = input_tensor.shape[0]
        h_t = self.hidden_state if self._memorize else np.zeros((1, self.hidden_size))

        self.input_tensor = input_tensor
        self.hidden_states.clear()

        self.hidden_activations.clear()
        self.fc_hidden_inputs.clear()
        self.fc_output_inputs.clear()

        outputs = []
        for t in range(time_steps):
            x_t = input_tensor[t:t+1]
            self.hidden_states.append(h_t)

            combined = np.hstack((x_t, h_t))

            h_pre = self.fc_hidden.forward(combined)

            self.fc_hidden_inputs.append(self.fc_hidden.input_tensor.copy())

            h_t = np.tanh(h_pre)
            self.hidden_activations.append(h_t)

            y_t = self.fc_output.forward(h_t)
            self.fc_output_inputs.append(self.fc_output.input_tensor.copy())
            outputs.append(y_t)

        self.hidden_state = h_t if self._memorize else np.zeros((1, self.hidden_size))
        return np.vstack(outputs)

    def backward(self, error_tensor):
        time_steps = error_tensor.shape[0]
        error_input = np.zeros((time_steps, self.input_size))

        dh_next = np.zeros((1, self.hidden_size))
        grad_hidden = np.zeros_like(self.fc_hidden.weights)
        grad_output = np.zeros_like(self.fc_output.weights)

        for t in reversed(range(time_steps)):
            dy = error_tensor[t:t+1]
            self.fc_output.input_tensor = self.fc_output_inputs[t]
            dh = self.fc_output.backward(dy)
            grad_output += self.fc_output.gradient_weights

            dh += dh_next
            h_t = self.hidden_activations[t]
            dtanh = dh * (1 - h_t ** 2)

            self.fc_hidden.input_tensor = self.fc_hidden_inputs[t]
            dcombined = self.fc_hidden.backward(dtanh)
            grad_hidden += self.fc_hidden.gradient_weights

            error_input[t:t+1] = dcombined[:, :self.input_size]
            dh_next = dcombined[:, self.input_size:]

        self.fc_hidden.gradient_weights = grad_hidden

        if self._optimizer:
            self.fc_hidden.weights = self._optimizer.calculate_update(self.fc_hidden.weights, grad_hidden)
            self.fc_output.weights = self._optimizer.calculate_update(self.fc_output.weights, grad_output)

        return error_input


