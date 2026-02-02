# Convolutional Neural Network (CNN)

A from-scratch implementation of Convolutional Neural Networks using only NumPy.

## Features

### Layers
- **Conv.py**: Convolutional layer with configurable kernel size, stride, and padding
- **Pooling.py**: Max and average pooling layers
- **Flatten.py**: Flattens multi-dimensional tensors for fully connected layers
- **FullyConnected.py**: Dense layer implementation
- **ReLU.py**: Rectified Linear Unit activation
- **SoftMax.py**: SoftMax activation for multi-class classification

### Optimization
- **Optimizers.py**: 
  - Stochastic Gradient Descent (SGD)
  - SGD with Momentum
  - Adam optimizer
- **Loss.py**: Cross-entropy loss implementation

### Network Architecture
- **NeuralNetwork.py**: Main network class supporting:
  - Layer stacking
  - Forward and backward propagation
  - Training and testing methods
  - Flexible optimizer and initializer configuration

## Usage

```python
from NeuralNetwork import NeuralNetwork
from Layers import Conv, ReLU, Pooling, Flatten, FullyConnected, SoftMax
from Optimization import Optimizers, Loss, Initializers

# Initialize
optimizer = Optimizers.Adam(learning_rate=0.001)
weight_init = Initializers.He()
bias_init = Initializers.Constant(0.1)

nn = NeuralNetwork(optimizer, weight_init, bias_init)

# Build architecture
nn.append_layer(Conv.Conv((1, 3, 3), 16, stride=1, padding=1))
nn.append_layer(ReLU.ReLU())
nn.append_layer(Pooling.Pooling((2, 2), stride=2))
nn.append_layer(Flatten.Flatten())
nn.append_layer(FullyConnected.FullyConnected(784, 128))
nn.append_layer(ReLU.ReLU())
nn.append_layer(FullyConnected.FullyConnected(128, 10))
nn.append_layer(SoftMax.SoftMax())

# Set data and loss
nn.data_layer = your_data_loader
nn.loss_layer = Loss.CrossEntropyLoss()

# Train
nn.train(iterations=10000)
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest NeuralNetworkTests.py -v
```

The test suite includes:
- Layer forward/backward pass validation
- Gradient checking
- Optimizer behavior tests
- Full network integration tests
- Performance benchmarks

## Implementation Details

### Convolution
- Supports arbitrary kernel sizes
- Configurable stride and padding
- Efficient implementation using im2col
- Proper gradient computation for backpropagation

### Pooling
- Max pooling with position tracking
- Average pooling
- Gradient routing for backpropagation

### Optimization
- Adam with bias correction
- Momentum with proper velocity updates
- Learning rate scheduling support

## Performance

All operations are vectorized using NumPy for efficiency while maintaining clarity of implementation.
