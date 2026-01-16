import copy
import pickle


def save(filename, net):
    """
    Save a neural network to a file using pickle.
    
    Args:
        filename: Path to save file
        net: NeuralNetwork object to save
    """
    with open(filename, 'wb') as f:
        pickle.dump(net, f)


def load(filename, data_layer):
    """
    Load a neural network from a file using pickle.
    
    Args:
        filename: Path to load file
        data_layer: Data layer to set in the loaded network
        
    Returns:
        Loaded NeuralNetwork object with data_layer set
    """
    with open(filename, 'rb') as f:
        net = pickle.load(f)
    net.data_layer = data_layer
    return net


class NeuralNetwork:
    def __init__(self, optimizer, weight_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self._phase = False

    @property
    def phase(self):
        return self._phase
    
    @phase.setter
    def phase(self, value):
        self._phase = value
        for layer in self.layers:
            layer.testing_phase = value

    def forward(self):
        out, self._last_y = self.data_layer.next()
        for layer in self.layers:
            out = layer.forward(out)

        loss_value = self.loss_layer.forward(out, self._last_y)
        
        # Add regularization loss from all trainable layers
        regularization_loss = 0
        for layer in self.layers:
            if getattr(layer, "trainable", False) and hasattr(layer, "optimizer") and layer.optimizer.regularizer is not None:
                regularization_loss += layer.optimizer.regularizer.norm(layer.weights)
        
        loss_value += regularization_loss
        return loss_value

    def backward(self):
        grad = self.loss_layer.backward(self._last_y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def append_layer(self, layer):
        if getattr(layer, "trainable", False):
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weight_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = False
        for _ in range(iterations):
            self.loss.append(self.forward()) # that runs recursively, and goes along the network 
            self.backward() # that also runs recursively, and propagates the error through the network 

    def test(self, input_tensor):
        self.phase = True
        out = input_tensor
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def __getstate__(self):
        """
        Prepare network for pickling by excluding the data_layer.
        
        Returns:
            Dictionary with network state, excluding data_layer
        """
        state = self.__dict__.copy()
        # Remove data_layer as it's a generator and cannot be pickled
        state['data_layer'] = None
        return state
    
    def __setstate__(self, state):
        """
        Restore network from pickled state.
        
        Args:
            state: Dictionary containing network state
        """
        self.__dict__.update(state)
        # data_layer should be None, will be set when loading
        self.data_layer = None
