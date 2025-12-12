import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.eps = np.finfo(float).eps

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor + np.finfo(float).eps

        log_probs = -label_tensor * np.log(self.prediction_tensor)

        loss = np.sum(log_probs)
        return loss

    def backward(self, label_tensor):
        error_tensor = -label_tensor / (self.prediction_tensor)
        return error_tensor