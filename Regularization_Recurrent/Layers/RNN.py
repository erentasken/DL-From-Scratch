from .Base import Base
from .FullyConnected import FullyConnected
import numpy as np

class RNN(Base):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Hidden state for the last timestep of previous sequence
        self.hidden_state = np.zeros((1, hidden_size))

        # Memorize flag for stateful sequences
        self._memorize = False

        # Optimizer for weight updates
        self._optimizer = None

        # Fully connected layers
        self.fc_hidden = FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_output = FullyConnected(hidden_size, output_size)

        # Store intermediate states for BPTT
        self.input_tensor = None
        self.hidden_states = []          # h_{t-1} for each timestep
        self.hidden_pre_activation = []  # pre-activation of hidden FC
        self.hidden_activations = []     # h_t after tanh
        self.fc_hidden_inputs = []       # inputs to hidden FC for backward
        self.fc_output_inputs = []       # inputs to output FC for backward

    # -----------------------------
    # Properties
    # -----------------------------
    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = bool(value)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
        # FC layers are decoupled from optimizer
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

    # -----------------------------
    # Initialization
    # -----------------------------
    def initialize(self, weight_initializer, bias_initializer):
        self.fc_hidden.initialize(weight_initializer, bias_initializer)
        self.fc_output.initialize(weight_initializer, bias_initializer)

    # -----------------------------
    # Forward pass
    # -----------------------------
    def forward(self, input_tensor):
        """
        Forward pass through the Elman RNN.
        input_tensor shape: (time_steps, input_size)
        Returns: (time_steps, output_size)
        """
        time_steps = input_tensor.shape[0]

        # Reset hidden state if not memorizing
        h_t = self.hidden_state if self._memorize else np.zeros((1, self.hidden_size))

        # Reset stored states
        self.input_tensor = input_tensor
        self.hidden_states.clear()
        self.hidden_pre_activation.clear()
        self.hidden_activations.clear()
        self.fc_hidden_inputs.clear()
        self.fc_output_inputs.clear()

        outputs = []

        for t in range(time_steps):
            x_t = input_tensor[t:t+1, :]  # keep shape (1, input_size)
            self.hidden_states.append(h_t)  # store previous hidden

            # Concatenate input and previous hidden
            combined = np.hstack((x_t, h_t))
            h_pre = self.fc_hidden.forward(combined)
            self.hidden_pre_activation.append(h_pre)
            self.fc_hidden_inputs.append(self.fc_hidden.input_tensor.copy())

            # Apply tanh
            h_t = np.tanh(h_pre)
            self.hidden_activations.append(h_t)

            # Compute output
            y_t = self.fc_output.forward(h_t)
            self.fc_output_inputs.append(self.fc_output.input_tensor.copy())
            outputs.append(y_t)

        # Store hidden state for next sequence if memorizing
        self.hidden_state = h_t if self._memorize else np.zeros((1, self.hidden_size))

        return np.vstack(outputs)

    # -----------------------------
    # Backward pass (BPTT)
    # -----------------------------
    def backward(self, error_tensor):
        """
        Backpropagation Through Time (BPTT)
        error_tensor shape: (time_steps, output_size)
        Returns: error tensor for previous layer (time_steps, input_size)
        """
        time_steps = error_tensor.shape[0]
        error_input = np.zeros((time_steps, self.input_size))

        dh_next = np.zeros((1, self.hidden_size))
        accumulated_gradient_hidden = np.zeros_like(self.fc_hidden.weights)
        accumulated_gradient_output = np.zeros_like(self.fc_output.weights)

        # Backprop through time
        for t in reversed(range(time_steps)):
            dy = error_tensor[t:t+1, :]
            
            # Output FC backward
            self.fc_output.input_tensor = self.fc_output_inputs[t]
            dh = self.fc_output.backward(dy)
            accumulated_gradient_output += self.fc_output.gradient_weights

            # Add gradient from next timestep
            dh += dh_next

            # Tanh backward
            h_t = self.hidden_activations[t]
            dtanh = dh * (1 - h_t ** 2)

            # Hidden FC backward
            self.fc_hidden.input_tensor = self.fc_hidden_inputs[t]
            dcombined = self.fc_hidden.backward(dtanh)
            accumulated_gradient_hidden += self.fc_hidden.gradient_weights

            # Split into dx and dh_prev
            dx = dcombined[:, :self.input_size]
            dh_next = dcombined[:, self.input_size:]

            error_input[t:t+1, :] = dx

        # Set hidden FC gradients
        self.fc_hidden.gradient_weights = accumulated_gradient_hidden

        # Apply optimizer updates if exists
        if self._optimizer is not None:
            self.fc_hidden.weights = self._optimizer.calculate_update(
                self.fc_hidden.weights, accumulated_gradient_hidden
            )
            self.fc_output.weights = self._optimizer.calculate_update(
                self.fc_output.weights, accumulated_gradient_output
            )

        return error_input

    # -----------------------------
    # Regularization
    # -----------------------------
    def calculate_regularization_loss(self):
        """Return sum of regularization losses for hidden and output weights"""
        if self._optimizer is not None and hasattr(self._optimizer, 'regularizer') \
           and self._optimizer.regularizer is not None:
            reg = self._optimizer.regularizer
            return reg.norm(self.fc_hidden.weights) + reg.norm(self.fc_output.weights)
        return 0.0
