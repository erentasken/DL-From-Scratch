# Feed-Forward Neural Network

A complete implementation of a multi-layer perceptron (MLP) from scratch using NumPy.

## Overview

This module implements a flexible feed-forward neural network with:
- Arbitrary number of layers
- Multiple activation functions
- Various optimization algorithms
- Comprehensive training and testing utilities

## Architecture

### Layers
- **FullyConnected.py**: Dense layer with weight matrix and bias vector
- **ReLU.py**: Rectified Linear Unit activation
- **SoftMax.py**: SoftMax activation for classification

### Optimization
- **Optimizers.py**:
  - SGD (Stochastic Gradient Descent)
  - SGD with Momentum
  - Adam optimizer
- **Loss.py**: Cross-entropy loss for classification tasks

## Quick Start

```python
from NeuralNetwork import NeuralNetwork
from Layers import FullyConnected, ReLU, SoftMax
from Optimization import Optimizers, Loss, Initializers

# Create network
optimizer = Optimizers.SGD(learning_rate=0.01)
weight_init = Initializers.He()
bias_init = Initializers.Constant(0.0)

nn = NeuralNetwork(optimizer, weight_init, bias_init)

# Build architecture (e.g., for MNIST)
nn.append_layer(FullyConnected.FullyConnected(784, 256))
nn.append_layer(ReLU.ReLU())
nn.append_layer(FullyConnected.FullyConnected(256, 128))
nn.append_layer(ReLU.ReLU())
nn.append_layer(FullyConnected.FullyConnected(128, 10))
nn.append_layer(SoftMax.SoftMax())

# Configure data and loss
nn.data_layer = data_loader
nn.loss_layer = Loss.CrossEntropyLoss()

# Train
nn.train(iterations=5000)

# Test
predictions = nn.test(test_data)
```

## Features

### Weight Initialization
- He initialization for ReLU networks
- Xavier initialization for sigmoid/tanh networks
- Constant initialization for biases

### Training
- Mini-batch gradient descent
- Automatic gradient computation
- Loss tracking for monitoring convergence

### Testing
- Forward pass for inference
- No gradient computation during testing

## Examples

See `main.ipynb` for detailed examples and visualizations.

## Testing

```bash
python -m pytest NeuralNetworkTests.py
```

## API Reference

### NeuralNetwork Class

**Methods:**
- `append_layer(layer)`: Add a layer to the network
- `train(iterations)`: Train for specified number of iterations
- `test(input_tensor)`: Run inference on input data
- `forward()`: Perform forward pass
- `backward()`: Perform backward pass

### Layer Base Class

All layers inherit from `Base.BaseLayer` and implement:
- `forward(input_tensor)`: Forward propagation
- `backward(error_tensor)`: Backward propagation

Trainable layers additionally implement:
- `initialize(weight_init, bias_init)`: Initialize parameters
