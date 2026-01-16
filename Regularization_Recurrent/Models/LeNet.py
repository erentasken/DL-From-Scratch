import sys
sys.path.insert(0, '..')

from Layers import Conv, Pooling, Flatten, FullyConnected, ReLU, SoftMax
from Optimization import Optimizers, Constraints, Loss
import NeuralNetwork


def build(learning_rate=5e-4, regularizer_strength=4e-4):
    """
    Build a LeNet-variant architecture for image classification.
    
    Architecture:
    - Conv layer: 1 input channel, 6 filters, 5x5 kernel, ReLU
    - Pooling: 2x2 max pool, stride 2
    - Conv layer: 6 input channels, 16 filters, 5x5 kernel, ReLU
    - Pooling: 2x2 max pool, stride 2
    - Flatten
    - Fully Connected: 256 -> 128, ReLU
    - Fully Connected: 128 -> 10, Softmax
    
    Args:
        learning_rate: Learning rate for ADAM optimizer (default: 5e-4)
        regularizer_strength: L2 regularization strength (default: 4e-4)
        
    Returns:
        NeuralNetwork object with LeNet architecture
    """
    # Create optimizer with L2 regularizer
    optimizer = Optimizers.Adam(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999
    )
    
    # Add L2 regularizer
    l2_regularizer = Constraints.L2_Regularizer(regularizer_strength)
    optimizer.add_regularizer(l2_regularizer)
    
    # Create initializers
    from Layers import Initializers
    weight_initializer = Initializers.He()
    bias_initializer = Initializers.Constant(0.0)
    
    # Create network
    net = NeuralNetwork.NeuralNetwork(optimizer, weight_initializer, bias_initializer)
    
    # Add layers
    # First convolutional block
    net.append_layer(Conv.Conv((1, 1), (1, 5, 5), 6))  # 1 input channel, 5x5 kernel, 6 filters
    net.append_layer(ReLU.ReLU())
    net.append_layer(Pooling.Pooling((2, 2), (2, 2)))  # 2x2 pool, stride 2
    
    # Second convolutional block
    net.append_layer(Conv.Conv((1, 1), (6, 5, 5), 16))  # 6 input channels, 5x5 kernel, 16 filters
    net.append_layer(ReLU.ReLU())
    net.append_layer(Pooling.Pooling((2, 2), (2, 2)))  # 2x2 pool, stride 2
    
    # Flatten
    net.append_layer(Flatten.Flatten())
    
    # Fully connected layers
    net.append_layer(FullyConnected.FullyConnected(784, 128))  # 16*7*7 = 784 -> 128
    net.append_layer(ReLU.ReLU())
    
    net.append_layer(FullyConnected.FullyConnected(128, 10))  # 128 -> 10 (MNIST classes)
    net.append_layer(SoftMax.SoftMax())
    
    # Set loss layer for training
    net.loss_layer = Loss.CrossEntropyLoss()
    
    return net
