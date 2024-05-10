import numpy as np

# This class includes all the activation functions
# Sintaxis:
# Normal function = class name (lowercase)
# Derivative function = class name (lowercase) + "_derivative"
# This sintaxis is used in Layer to get the functions properly
class ActFunction:
    
    # Sigmoid functions
    class Sigmoid:
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(x):
            return x * (1 - x)
    
    # ReLu functions
    class ReLu:
        def relu(x):
            return x * (x > 0)

        def relu_derivative(x):
            if isinstance(x, (float, int)):
                if x > 0:
                    return 1
                else:
                    return 0
            elif isinstance(x, np.ndarray):
                return np.where(x > 0, 1, 0)
            else:
                raise ValueError("Input must be a float, int, or numpy array.")
    
    # Tanh functions
    class Tanh:
        def tanh(x):
            return np.tanh(x)
        
        def tanh_derivative(x):
            return 1 - np.tanh(x)**2
    
    # Sgn functions
    class Sgn:
        def sgn(x):
            return np.where(x >= 0, 1, -1)

        def sgn_derivative(x):
            return np.where(x != 0, 0, np.nan)
