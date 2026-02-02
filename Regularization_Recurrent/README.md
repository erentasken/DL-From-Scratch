# Regularization & Recurrent Neural Networks

Advanced neural network implementations featuring regularization techniques and recurrent architectures.

## Overview

This module extends the basic neural network implementations with:
- Dropout regularization
- L2 weight regularization
- Recurrent Neural Network (RNN) layers
- LSTM (Long Short-Term Memory) layers
- LeNet architecture for image classification
- Advanced training utilities

## Features

### Regularization Techniques

**Dropout**
- Randomly drops neurons during training
- Prevents overfitting
- Configurable dropout rate

**L2 Regularization**
- Weight decay
- Prevents large weights
- Configurable regularization strength

### Recurrent Layers

**RNN (Recurrent Neural Network)**
- Basic recurrent layer
- Handles sequential data
- Backpropagation through time (BPTT)

**LSTM (Long Short-Term Memory)**
- Advanced RNN with gates
- Solves vanishing gradient problem
- Memory cell for long-term dependencies

### Model Architectures

**LeNet**
- Classic CNN architecture
- Designed for digit recognition
- Efficient and accurate

## Project Structure

```
Regularization_Recurrent/
├── Layers/              # Layer implementations with regularization
├── Optimization/        # Optimizers with L2 regularization
├── Models/             # Pre-built model architectures
├── Data/               # Dataset loaders and utilities
├── NeuralNetwork.py    # Enhanced neural network class
├── TrainLeNet.py       # Training script for LeNet
└── trained/            # Saved model checkpoints
```

## Usage

### Training LeNet

```bash
python TrainLeNet.py
```

### Using Dropout

```python
from Layers import FullyConnected, Dropout

# Add dropout layer (50% dropout rate)
nn.append_layer(FullyConnected.FullyConnected(256, 128))
nn.append_layer(Dropout.Dropout(0.5))
nn.append_layer(ReLU.ReLU())
```

### L2 Regularization

```python
from Optimization import Optimizers

# Create optimizer with L2 regularization
optimizer = Optimizers.Adam(
    learning_rate=0.001,
    regularizer={'type': 'L2', 'lambda': 0.0001}
)
```

### RNN for Sequential Data

```python
from Layers import RNN

# Add RNN layer
# input_size: feature dimension
# hidden_size: hidden state dimension
rnn_layer = RNN.RNN(input_size=10, hidden_size=64)
nn.append_layer(rnn_layer)
```

### LSTM Layer

```python
from Layers import LSTM

# Add LSTM layer
lstm_layer = LSTM.LSTM(input_size=10, hidden_size=64)
nn.append_layer(lstm_layer)
```

## Training LeNet on MNIST

```python
from Models import LeNet
from Optimization import Optimizers, Loss

# Create LeNet model
model = LeNet.LeNet()

# Configure optimizer with regularization
optimizer = Optimizers.Adam(
    learning_rate=0.001,
    regularizer={'type': 'L2', 'lambda': 0.0005}
)

# Set up training
model.optimizer = optimizer
model.loss_layer = Loss.CrossEntropyLoss()

# Train
model.train(epochs=10, batch_size=64)

# Save model
model.save('trained/lenet_mnist.pkl')
```

## Testing

```bash
python -m pytest NeuralNetworkTests.py -v
```

## Advanced Features

### Learning Rate Scheduling
```python
# Decay learning rate over time
for epoch in range(num_epochs):
    if epoch % 10 == 0:
        optimizer.learning_rate *= 0.9
    nn.train(iterations_per_epoch)
```

### Early Stopping
```python
# Monitor validation loss
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = nn.train(train_iterations)
    val_loss = evaluate(nn, val_data)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model(nn, 'best_model.pkl')
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping triggered")
        break
```

### Gradient Clipping
```python
# Prevent exploding gradients in RNNs
def clip_gradients(gradient, threshold=5.0):
    norm = np.linalg.norm(gradient)
    if norm > threshold:
        gradient = gradient * (threshold / norm)
    return gradient
```

## Performance Considerations

- Dropout only active during training
- BPTT can be memory intensive for long sequences
- Use truncated BPTT for very long sequences
- Batch normalization can help with regularization

## Requirements

```
numpy==1.26.4
matplotlib
scipy
scikit-learn==1.1.3
scikit-image
tabulate
```

## References

- Dropout: Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014)
- LSTM: Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)
- LeNet: LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998)
