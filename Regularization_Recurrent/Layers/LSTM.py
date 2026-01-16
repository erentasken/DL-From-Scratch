from .Base import Base
from .FullyConnected import FullyConnected
import numpy as np

class LSTM(Base):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Hidden and cell states for the previous sequence
        self.hidden_state = np.zeros((1, hidden_size))
        self.cell_state = np.zeros((1, hidden_size))

        # Memorize flag for stateful sequences
        self._memorize = False

        # Optimizer
        self._optimizer = None

        # Fully connected layers for gates: input, forget, output, candidate
        self.fc_i = FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_f = FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_o = FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_g = FullyConnected(input_size + hidden_size, hidden_size)

        # Fully connected layer for output
        self.fc_output = FullyConnected(hidden_size, output_size)

        # Storage for forward/backward
        self.input_tensor = None
        self.hiddens = []         # h_t
        self.cells = []           # c_t
        self.gates = []           # stores i, f, o, g pre-activations
        self.fc_inputs = []       # concatenated x_t and h_{t-1} per timestep
        self.fc_output_inputs = []

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
        # Decouple FC layers from optimizer
        self.fc_i.optimizer = None
        self.fc_f.optimizer = None
        self.fc_o.optimizer = None
        self.fc_g.optimizer = None
        self.fc_output.optimizer = None

    @property
    def weights(self):
        # Return gate weights including bias rows
        return np.hstack([self.fc_i.weights,
                        self.fc_f.weights,
                        self.fc_o.weights,
                        self.fc_g.weights])

    @weights.setter
    def weights(self, value):
        hidden = self.hidden_size
        input_hidden = self.input_size + self.hidden_size
        
        # Split the weights (expecting bias rows included) and assign directly
        i_weights = value[:, :hidden]
        f_weights = value[:, hidden:2*hidden]
        o_weights = value[:, 2*hidden:3*hidden]
        g_weights = value[:, 3*hidden:]

        self.fc_i.weights = i_weights
        self.fc_f.weights = f_weights
        self.fc_o.weights = o_weights
        self.fc_g.weights = g_weights

    @property
    def gradient_weights(self):
        # Return gate gradients including bias rows
        return np.hstack([self.fc_i.gradient_weights,
                         self.fc_f.gradient_weights,
                         self.fc_o.gradient_weights,
                         self.fc_g.gradient_weights])

    @gradient_weights.setter
    def gradient_weights(self, value):
        hidden = self.hidden_size
        input_hidden = self.input_size + self.hidden_size
        
        # Split the gradients (they include bias rows) and assign directly
        split = np.split(value, 4, axis=1)
        i_grad, f_grad, o_grad, g_grad = split

        self.fc_i.gradient_weights = i_grad
        self.fc_f.gradient_weights = f_grad
        self.fc_o.gradient_weights = o_grad
        self.fc_g.gradient_weights = g_grad

    # -----------------------------
    # Initialization
    # -----------------------------
    def initialize(self, weight_initializer, bias_initializer):
        for fc in [self.fc_i, self.fc_f, self.fc_o, self.fc_g, self.fc_output]:
            fc.initialize(weight_initializer, bias_initializer)

    # -----------------------------
    # Forward pass
    # -----------------------------
    def forward(self, input_tensor):
        """
        Forward pass through LSTM.
        input_tensor shape: (time_steps, input_size)
        Returns: (time_steps, output_size)
        """
        time_steps = input_tensor.shape[0]

        # Initialize hidden/cell states
        if self._memorize:
            h_t = self.hidden_state
            c_t = self.cell_state
        else:
            h_t = np.zeros((1, self.hidden_size))
            c_t = np.zeros((1, self.hidden_size))

        # Clear stored states
        self.input_tensor = input_tensor
        self.hiddens.clear()
        self.cells.clear()
        self.gates.clear()
        self.fc_inputs.clear()
        self.fc_output_inputs.clear()

        outputs = []

        for t in range(time_steps):
            x_t = input_tensor[t:t+1, :]  # (1, input_size)
            combined = np.hstack((x_t, h_t))  # (1, input + hidden)
            # store the input with bias for FC backward (batch, input+hidden+1)
            combined_with_bias = np.hstack((combined, np.ones((combined.shape[0], 1))))
            self.fc_inputs.append(combined_with_bias.copy())

            # Compute gates (FC.forward will also set their own input_tensor)
            i_t = self._sigmoid(self.fc_i.forward(combined))
            f_t = self._sigmoid(self.fc_f.forward(combined))
            o_t = self._sigmoid(self.fc_o.forward(combined))
            g_t = np.tanh(self.fc_g.forward(combined))
            self.gates.append((i_t, f_t, o_t, g_t))

            # Update cell and hidden state
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * np.tanh(c_t)

            self.cells.append(c_t.copy())
            self.hiddens.append(h_t.copy())

            # Output layer
            y_t = self.fc_output.forward(h_t)
            # store the FC input tensor (including bias) for backward
            self.fc_output_inputs.append(self.fc_output.input_tensor.copy())
            outputs.append(y_t)

        # Store last hidden/cell state for memorization
        if self._memorize:
            self.hidden_state = h_t
            self.cell_state = c_t

        return np.vstack(outputs)

    def backward(self, error_tensor):
        """
        Backpropagation through time for LSTM.
        error_tensor shape: (time_steps, output_size)
        Returns: error tensor for previous layer (time_steps, input_size)
        """
        time_steps = error_tensor.shape[0]
        error_input = np.zeros((time_steps, self.input_size))

        dh_next = np.zeros((1, self.hidden_size))
        dc_next = np.zeros((1, self.hidden_size))

        # Initialize accumulated gradients with correct shapes (use weights shapes)
        accumulated_grads = {
            'i': np.zeros_like(self.fc_i.weights),
            'f': np.zeros_like(self.fc_f.weights),
            'o': np.zeros_like(self.fc_o.weights),
            'g': np.zeros_like(self.fc_g.weights),
            'out': np.zeros_like(self.fc_output.weights)
        }

        for t in reversed(range(time_steps)):
            dy = error_tensor[t:t+1, :]

            # ---------------- Output layer backward ----------------
            # Set the input tensor that was used in forward pass
            self.fc_output.input_tensor = self.fc_output_inputs[t]
            
            # Perform backward pass
            dh = self.fc_output.backward(dy)
            
            # Accumulate gradients
            accumulated_grads['out'] += self.fc_output.gradient_weights

            dh += dh_next  # add gradient from next timestep

            # ---------------- LSTM gates ----------------
            i_t, f_t, o_t, g_t = self.gates[t]
            c_t = self.cells[t]
            c_prev = self.cells[t-1] if t > 0 else np.zeros((1, self.hidden_size))

            # Backprop through cell
            do = dh * np.tanh(c_t)
            dc = dh * o_t * (1 - np.tanh(c_t)**2) + dc_next
            di = dc * g_t
            dg = dc * i_t
            df = dc * c_prev
            dc_next = dc * f_t

            # Apply gate activation derivatives
            di *= i_t * (1 - i_t)
            df *= f_t * (1 - f_t)
            do *= o_t * (1 - o_t)
            dg *= 1 - g_t ** 2

            # ---------------- Backprop through FC layers for gates ----------------
            # Input gate
            self.fc_i.input_tensor = self.fc_inputs[t]
            d_comb_i = self.fc_i.backward(di)
            accumulated_grads['i'] += self.fc_i.gradient_weights

            # Forget gate
            self.fc_f.input_tensor = self.fc_inputs[t]
            d_comb_f = self.fc_f.backward(df)
            accumulated_grads['f'] += self.fc_f.gradient_weights

            # Output gate
            self.fc_o.input_tensor = self.fc_inputs[t]
            d_comb_o = self.fc_o.backward(do)
            accumulated_grads['o'] += self.fc_o.gradient_weights

            # Candidate gate
            self.fc_g.input_tensor = self.fc_inputs[t]
            d_comb_g = self.fc_g.backward(dg)
            accumulated_grads['g'] += self.fc_g.gradient_weights

            # ---------------- Gradients w.r.t input ----------------
            # Sum gradients from all gates for input x
            dx = d_comb_i[:, :self.input_size] + d_comb_f[:, :self.input_size] + \
                d_comb_o[:, :self.input_size] + d_comb_g[:, :self.input_size]
            
            # Sum gradients from all gates for hidden state
            dh_next = d_comb_i[:, self.input_size:] + d_comb_f[:, self.input_size:] + \
                    d_comb_o[:, self.input_size:] + d_comb_g[:, self.input_size:]

            error_input[t:t+1, :] = dx

        # ---------------- Set accumulated gradients ----------------
        self.fc_i.gradient_weights = accumulated_grads['i']
        self.fc_f.gradient_weights = accumulated_grads['f']
        self.fc_o.gradient_weights = accumulated_grads['o']
        self.fc_g.gradient_weights = accumulated_grads['g']
        self.fc_output.gradient_weights = accumulated_grads['out']

        # ---------------- Apply optimizer updates ----------------
        if self._optimizer is not None:
            # Update weights for each FC layer
            self.fc_i.weights = self._optimizer.calculate_update(self.fc_i.weights, self.fc_i.gradient_weights)
            self.fc_f.weights = self._optimizer.calculate_update(self.fc_f.weights, self.fc_f.gradient_weights)
            self.fc_o.weights = self._optimizer.calculate_update(self.fc_o.weights, self.fc_o.gradient_weights)
            self.fc_g.weights = self._optimizer.calculate_update(self.fc_g.weights, self.fc_g.gradient_weights)
            self.fc_output.weights = self._optimizer.calculate_update(self.fc_output.weights, self.fc_output.gradient_weights)

        return error_input


    # -----------------------------
    # Regularization
    # -----------------------------
    def calculate_regularization_loss(self):
        if self._optimizer is not None and hasattr(self._optimizer, 'regularizer') \
           and self._optimizer.regularizer is not None:
            reg = self._optimizer.regularizer
            return sum([reg.norm(fc.weights) for fc in 
                        [self.fc_i, self.fc_f, self.fc_o, self.fc_g, self.fc_output]])
        return 0.0

    # -----------------------------
    # Utility
    # -----------------------------
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def reset_state(self):
        """Reset hidden and cell states to zeros"""
        self.hidden_state = np.zeros((1, self.hidden_size))
        self.cell_state = np.zeros((1, self.hidden_size))