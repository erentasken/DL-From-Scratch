# NumPy Pattern Generation and Manipulation

Educational exercises focusing on NumPy array operations, pattern generation, and data manipulation techniques.

## Overview

This module contains implementations of various pattern generators and data manipulation utilities built entirely with NumPy. Designed to develop proficiency in:
- NumPy broadcasting
- Array indexing and slicing
- Vectorized operations
- Pattern generation algorithms

## Modules

### pattern.py
Pattern generation classes for creating various geometric patterns:

**Checker**: Generates checkerboard patterns
```python
import pattern
c = pattern.Checker(resolution=250, tile_size=25)
c.draw()
c.show()  # Display the pattern
```

**Circle**: Creates circular patterns
```python
c = pattern.Circle(resolution=512, radius=100, position=(256, 256))
c.draw()
c.show()
```

**Spectrum**: Generates color spectrum patterns
```python
s = pattern.Spectrum(resolution=256)
s.draw()
s.show()
```

### generator.py
Data generator utilities for creating synthetic datasets:
- Random pattern generation
- Data augmentation utilities
- Batch generation for training

### Data
- **exercise_data/**: 100 NumPy arrays (0.npy - 99.npy) for testing
- **reference_arrays/**: Reference patterns for validation
- **Labels.json**: Metadata and labels for datasets

## Usage Examples

### Creating a Checkerboard
```python
from pattern import Checker
import matplotlib.pyplot as plt

# Create 500x500 checkerboard with 50px tiles
checker = Checker(500, 50)
checker.draw()

# Display
plt.imshow(checker.output, cmap='gray')
plt.show()
```

### Working with Exercise Data
```python
import numpy as np

# Load data
data = np.load('exercise_data/0.npy')
print(f"Shape: {data.shape}")
print(f"Data type: {data.dtype}")

# Process...
```

## Testing

Run comprehensive tests to verify implementations:

```bash
python -m pytest NumpyTests.py -v
```

Test coverage includes:
- Pattern generation correctness
- Shape and dimension validation
- Edge case handling
- Performance benchmarks

## Key Concepts Demonstrated

### Broadcasting
Efficient array operations without explicit loops:
```python
# Create grid coordinates
x = np.arange(resolution)
y = np.arange(resolution)
X, Y = np.meshgrid(x, y)
```

### Vectorization
Replace loops with vectorized operations:
```python
# Vectorized distance calculation
distances = np.sqrt((X - cx)**2 + (Y - cy)**2)
circle = distances < radius
```

### Indexing
Advanced indexing techniques:
```python
# Boolean indexing
pattern[mask] = 1

# Fancy indexing
selected = data[indices]
```

## Requirements

```
numpy>=1.26.4
matplotlib
scipy
```

## Learning Path

1. Start with `pattern.py` to understand basic pattern generation
2. Explore `generator.py` for data generation techniques
3. Study the test cases in `NumpyTests.py` for best practices
4. Experiment with `main.ipynb` for interactive learning

## Performance Tips

- Avoid Python loops; use NumPy operations
- Preallocate arrays when possible
- Use appropriate data types (uint8 for images)
- Leverage broadcasting instead of explicit expansion
- Profile code to identify bottlenecks
