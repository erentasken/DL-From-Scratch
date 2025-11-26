import copy

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None


    def forward(self):
        out, self._last_y = self.data_layer.next()
        for layer in self.layers:
            out = layer.forward(out)

        loss_value = self.loss_layer.forward(out, self._last_y)
        return loss_value

    def backward(self):
        grad = self.loss_layer.backward(self._last_y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def append_layer(self, layer):
        if getattr(layer, "trainable", False):
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            self.loss.append(self.forward()) # that runs recursively, and goes along the network 
            self.backward() # that also runs recursively, and propagates the error through the network 

    def test(self, input_tensor):
        out = input_tensor
        for layer in self.layers:
            out = layer.forward(out)
        return out
