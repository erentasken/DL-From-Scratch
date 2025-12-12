import numpy as np
class Adam:
    def __init__(self, learning_rate, beta1, beta2, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def calculate_update(self, weights, gradients):
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        self.t += 1

        # first moment ( momentum )
        # if the gradients keeps growing in the same direction , m grows, step is faster 
        # if it changes the direction, m smooths the oscillation
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients 

        #second moment ( uncentered variance )
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)  

        # bias-corrected moments
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # update weights
        weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return weights

class SgdWithMomentum:
    def __init__(self, learning_rate, momentum):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def calculate_update(self, weights, gradients):
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)

        # update velocity
        self.velocity = self.momentum * self.velocity + self.lr * gradients

        # update weights
        weights -= self.velocity
        return weights
    
class Sgd:
    learning_rate = None
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return weight_tensor