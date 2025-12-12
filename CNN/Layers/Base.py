class Base:
    trainable = None
    def __init__(self):
        self.trainable = False
    def forward(input_tensor):
        pass
    def backward(error_tensor):
        pass