import numpy as np
from .Base import Base
from .FullyConnected import FullyConnected
from .Sigmoid import Sigmoid
class LSTM(Base):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size
        self.hidden_state = np.zeros((1, hidden_size))
        self.cell_state = np.zeros((1, hidden_size))
        self._memorize = False
        self._optimizer = None
        self.trainable = True

        self._sigmoid = Sigmoid().forward

        # Fully connected layers for gates and output
        self.fc_i = FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_f = FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_o = FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_g = FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_output = FullyConnected(hidden_size, output_size)

        # Stored states for BPTT
        self.hiddens, self.cells, self.gates = [], [], []
        self.fc_inputs, self.fc_output_inputs = [], []

    @property
    def memorize(self): return self._memorize
    @memorize.setter
    def memorize(self, val): self._memorize = bool(val)

    @property
    def optimizer(self): return self._optimizer
    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
        for fc in [self.fc_i, self.fc_f, self.fc_o, self.fc_g, self.fc_output]:
            fc.optimizer = None

    def initialize(self, weight_init, bias_init):
        for fc in [self.fc_i, self.fc_f, self.fc_o, self.fc_g, self.fc_output]:
            fc.initialize(weight_init, bias_init)



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

    def forward(self, x):
        time_steps = x.shape[0]
        h_t = self.hidden_state if self._memorize else np.zeros((1, self.hidden_size))
        c_t = self.cell_state if self._memorize else np.zeros((1, self.hidden_size))

        self.hiddens.clear(); self.cells.clear()
        self.gates.clear(); self.fc_inputs.clear(); self.fc_output_inputs.clear()
        outputs = []

        for t in range(time_steps):
            x_t = x[t:t+1]
            combined = np.hstack((x_t, h_t))
            self.fc_inputs.append(np.hstack((combined, np.ones((1,1)))).copy())
            
            i_t, f_t, o_t, g_t = (self._sigmoid(self.fc_i.forward(combined)),
                                  self._sigmoid(self.fc_f.forward(combined)),
                                  self._sigmoid(self.fc_o.forward(combined)),
                                  np.tanh(self.fc_g.forward(combined)))
            self.gates.append((i_t, f_t, o_t, g_t))

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * np.tanh(c_t)
            self.cells.append(c_t.copy())
            self.hiddens.append(h_t.copy())

            y_t = self.fc_output.forward(h_t)
            self.fc_output_inputs.append(self.fc_output.input_tensor.copy())
            outputs.append(y_t)

        if self._memorize: self.hidden_state, self.cell_state = h_t, c_t
        return np.vstack(outputs)

    def backward(self, error_tensor):
        time_steps = error_tensor.shape[0]
        error_input = np.zeros((time_steps, self.input_size))
        dh_next, dc_next = np.zeros((1, self.hidden_size)), np.zeros((1, self.hidden_size))
        accumulated_grads = {k: np.zeros_like(fc.weights) for k, fc in
                             zip(['i','f','o','g','out'],
                                 [self.fc_i, self.fc_f, self.fc_o, self.fc_g, self.fc_output])}

        for t in reversed(range(time_steps)):
            dy = error_tensor[t:t+1]
            self.fc_output.input_tensor = self.fc_output_inputs[t]
            dh = self.fc_output.backward(dy)
            accumulated_grads['out'] += self.fc_output.gradient_weights
            dh += dh_next

            i_t, f_t, o_t, g_t = self.gates[t]
            c_t = self.cells[t]; c_prev = self.cells[t-1] if t>0 else np.zeros((1,self.hidden_size))

            do = dh * np.tanh(c_t); dc = dh * o_t * (1 - np.tanh(c_t)**2) + dc_next
            di, dg, df = dc * g_t, dc * i_t, dc * c_prev
            dc_next = dc * f_t

            di *= i_t * (1-i_t); df *= f_t * (1-f_t); do *= o_t * (1-o_t); dg *= 1 - g_t**2

            # Gate FC backward
            for gate, d_gate in zip(['i','f','o','g'], [di, df, do, dg]):
                fc = getattr(self, f'fc_{gate}')
                fc.input_tensor = self.fc_inputs[t]
                d_comb = fc.backward(d_gate)
                accumulated_grads[gate] += fc.gradient_weights
                if gate=='i': dx, dh_next_partial = d_comb[:, :self.input_size], d_comb[:, self.input_size:]
                else:
                    dx += d_comb[:, :self.input_size]; dh_next_partial += d_comb[:, self.input_size:]
            dh_next = dh_next_partial
            error_input[t:t+1] = dx

        # Set accumulated gradients
        self.fc_i.gradient_weights = accumulated_grads['i']
        self.fc_f.gradient_weights = accumulated_grads['f']
        self.fc_o.gradient_weights = accumulated_grads['o']
        self.fc_g.gradient_weights = accumulated_grads['g']
        self.fc_output.gradient_weights = accumulated_grads['out']

        if self._optimizer:
            for fc in [self.fc_i, self.fc_f, self.fc_o, self.fc_g, self.fc_output]:
                fc.weights = self._optimizer.calculate_update(fc.weights, fc.gradient_weights)

        return error_input