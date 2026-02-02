# Deep Learning Exercises

A collection of deep learning implementations built from scratch using NumPy, including Convolutional Neural Networks (CNN), Feed-Forward Neural Networks, and Recurrent Neural Networks with regularization techniques.

## 📚 Project Overview

This repository contains implementations of various neural network architectures and deep learning concepts, built as educational exercises. Each component is implemented from scratch to provide a deep understanding of the underlying mathematics and algorithms.

## 🗂️ Project Structure

```
.
├── CNN/                              # Convolutional Neural Network implementation
│   ├── Layers/                       # CNN layer implementations
│   │   ├── Conv.py                   # Convolutional layer
│   │   ├── Pooling.py                # Pooling layer
│   │   ├── Flatten.py                # Flatten layer
│   │   ├── FullyConnected.py         # Fully connected layer
│   │   ├── ReLU.py                   # ReLU activation
│   │   └── SoftMax.py                # SoftMax activation
│   ├── Optimization/                 # Optimization algorithms
│   │   ├── Optimizers.py             # SGD, Adam, etc.
│   │   └── Loss.py                   # Loss functions
│   ├── NeuralNetwork.py              # Main CNN architecture
│   └── NeuralNetworkTests.py         # Comprehensive test suite
│
├── FeedForwardNeuralNetwork/         # Feed-forward neural network
│   ├── Layers/                       # Layer implementations
│   ├── Optimization/                 # Optimizers and loss functions
│   ├── NeuralNetwork.py              # Main network class
│   ├── NeuralNetworkTests.py         # Test suite
│   └── main.ipynb                    # Example usage notebook
│
├── Regularization_Recurrent/         # Recurrent networks with regularization
│   ├── Layers/                       # RNN layer implementations
│   ├── Optimization/                 # Optimizers with regularization
│   ├── Models/                       # Model architectures
│   ├── Data/                         # Dataset utilities
│   └── TrainLeNet.py                 # Training script
│
├── numpy/                            # NumPy pattern generation exercises
│   ├── pattern.py                    # Pattern generators
│   ├── generator.py                  # Data generators
│   ├── NumpyTests.py                 # Unit tests
│   └── exercise_data/                # Exercise datasets
│
└── playground.ipynb                  # Experimental notebook
```

## 🚀 Features

### CNN Implementation
- **Convolutional Layers**: Forward and backward propagation with stride and padding support
- **Pooling Layers**: Max and average pooling with configurable kernel sizes
- **Activation Functions**: ReLU, SoftMax, and more
- **Weight Initialization**: He, Xavier, and custom initializers
- **Optimizers**: SGD, SGD with Momentum, Adam
- **Comprehensive Testing**: Over 1000 lines of unit tests

### Feed-Forward Neural Network
- Multi-layer perceptron implementation
- Flexible architecture configuration
- Various activation functions
- Backpropagation from scratch
- Training and testing utilities

### Regularization & Recurrent Networks
- Dropout and L2 regularization
- Recurrent neural network layers
- LeNet architecture implementation
- Advanced optimization techniques

### NumPy Exercises
- Pattern generation (checkerboard, circles, etc.)
- Data manipulation and transformation
- Array operations and broadcasting
- Comprehensive test suite

## 📋 Requirements

```
numpy==1.26.4
matplotlib
scipy
scikit-learn
scikit-image
tabulate
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deep-learning-exercises.git
cd deep-learning-exercises
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### CNN Example

```python
from CNN.NeuralNetwork import NeuralNetwork
from CNN.Layers import Conv, ReLU, Pooling, Flatten, FullyConnected, SoftMax
from CNN.Optimization import Optimizers, Loss

# Initialize network
optimizer = Optimizers.Adam(0.001, 0.9, 0.999)
nn = NeuralNetwork(optimizer, weight_initializer, bias_initializer)

# Build architecture
nn.append_layer(Conv.Conv((1, 5, 5), 3, stride=1, padding=2))
nn.append_layer(ReLU.ReLU())
nn.append_layer(Pooling.Pooling((2, 2), stride=2))
nn.append_layer(Flatten.Flatten())
nn.append_layer(FullyConnected.FullyConnected(128, 10))
nn.append_layer(SoftMax.SoftMax())

# Set loss and train
nn.loss_layer = Loss.CrossEntropyLoss()
nn.train(iterations=1000)
```

### Feed-Forward Network Example

```python
from FeedForwardNeuralNetwork.NeuralNetwork import NeuralNetwork
from FeedForwardNeuralNetwork.Layers import FullyConnected, ReLU, SoftMax
from FeedForwardNeuralNetwork.Optimization import Optimizers, Loss

# Create network
nn = NeuralNetwork(optimizer, weight_initializer, bias_initializer)

# Add layers
nn.append_layer(FullyConnected.FullyConnected(784, 256))
nn.append_layer(ReLU.ReLU())
nn.append_layer(FullyConnected.FullyConnected(256, 10))
nn.append_layer(SoftMax.SoftMax())

# Train
nn.loss_layer = Loss.CrossEntropyLoss()
nn.train(iterations=5000)
```

## 🧪 Running Tests

Each module includes comprehensive unit tests:

```bash
# Test CNN implementation
cd CNN
python -m pytest NeuralNetworkTests.py

# Test Feed-Forward Network
cd FeedForwardNeuralNetwork
python -m pytest NeuralNetworkTests.py

# Test NumPy exercises
cd numpy
python -m pytest NumpyTests.py
```

## 📊 Project Highlights

- **From Scratch Implementation**: All components built without high-level frameworks (TensorFlow/PyTorch)
- **Educational Focus**: Clear, readable code with detailed comments
- **Comprehensive Testing**: Extensive test suites ensure correctness
- **Modular Design**: Easily extensible architecture
- **Mathematical Rigor**: Proper implementation of gradients and optimization algorithms

## 🎯 Learning Objectives

This project demonstrates understanding of:
- Neural network architectures and forward/backward propagation
- Convolutional operations and their mathematical foundations
- Optimization algorithms (SGD, Momentum, Adam)
- Weight initialization strategies
- Activation functions and their derivatives
- Loss functions and gradient computation
- NumPy broadcasting and vectorization
- Software engineering best practices for ML projects

## 🤝 Contributing

This is an educational project, but suggestions and improvements are welcome! Feel free to:
- Report bugs
- Suggest enhancements
- Submit pull requests
- Improve documentation

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Eren**

## 🙏 Acknowledgments

- Built as part of a Deep Learning course curriculum
- Inspired by various deep learning textbooks and papers
- Test cases provided by course instructors

## 📚 Resources

- [Deep Learning Book by Ian Goodfellow](https://www.deeplearningbook.org/)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Neural Networks and Deep Learning by Michael Nielsen](http://neuralnetworksanddeeplearning.com/)

---

⭐ If you find this repository helpful, please consider giving it a star!
