# Contributing to Deep Learning Exercises

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Relevant code snippets or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:
- Check if the enhancement has already been suggested
- Provide a clear description of the feature
- Explain why it would be useful
- Include code examples if applicable

### Pull Requests

1. **Fork the repository**

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add docstrings to new functions/classes
   - Update tests if needed
   - Update documentation

4. **Test your changes**
   ```bash
   # Run all tests
   pytest
   
   # Run specific module tests
   cd CNN
   pytest NeuralNetworkTests.py -v
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add: Brief description of changes"
   ```
   
   Use conventional commit messages:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Update:` for changes to existing features
   - `Docs:` for documentation changes
   - `Test:` for test additions/modifications

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure all tests pass

## Code Style Guidelines

### Python Style
- Follow PEP 8 style guide
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use descriptive variable names

### Documentation
- Add docstrings to all public functions and classes
- Use numpy-style docstrings:
  ```python
  def forward(self, input_tensor):
      """
      Perform forward propagation.
      
      Parameters
      ----------
      input_tensor : np.ndarray
          Input data of shape (batch_size, input_dim)
      
      Returns
      -------
      np.ndarray
          Output of shape (batch_size, output_dim)
      """
      pass
  ```

### Testing
- Write tests for new features
- Maintain or improve test coverage
- Use descriptive test names
- Test edge cases

### Example Code Structure
```python
import numpy as np
from .Base import BaseLayer


class MyLayer(BaseLayer):
    """
    Brief description of the layer.
    
    Attributes
    ----------
    param_name : type
        Description of parameter
    """
    
    def __init__(self, param1, param2):
        """Initialize the layer."""
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def forward(self, input_tensor):
        """Forward propagation."""
        # Implementation
        return output
    
    def backward(self, error_tensor):
        """Backward propagation."""
        # Implementation
        return gradient
```

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/deep-learning-exercises.git
   cd deep-learning-exercises
   ```

2. Run setup script:
   ```bash
   ./setup.sh
   ```

3. Activate virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. Install development dependencies:
   ```bash
   pip install pytest pytest-cov ipython jupyter
   ```

## Testing Guidelines

### Running Tests
```bash
# All tests
pytest

# Specific module
pytest CNN/NeuralNetworkTests.py -v

# With coverage
pytest --cov=CNN --cov-report=html
```

### Writing Tests
```python
import unittest
import numpy as np

class TestMyLayer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.layer = MyLayer(param1, param2)
        self.input = np.random.rand(10, 5)
    
    def test_forward_shape(self):
        """Test that forward pass returns correct shape."""
        output = self.layer.forward(self.input)
        self.assertEqual(output.shape, (10, 5))
    
    def test_backward_gradient(self):
        """Test gradient computation."""
        # Test implementation
        pass
```

## Project Structure

When adding new files, follow the existing structure:
```
Module/
├── Layers/
│   ├── __init__.py
│   ├── Base.py
│   └── YourLayer.py
├── Optimization/
│   ├── __init__.py
│   └── YourOptimizer.py
├── NeuralNetwork.py
├── Tests.py
└── README.md
```

## Questions?

Feel free to open an issue for:
- Questions about the codebase
- Clarification on contribution guidelines
- Discussion of potential features

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on what is best for the project
- Show empathy towards other community members

## Recognition

Contributors will be acknowledged in the project README.

Thank you for contributing! 🎉
