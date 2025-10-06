import numpy as np

# Activation functions
def _relu(x, device, alpha):
    return device.maximum(0, x)
def _leaky_relu(x, device, alpha):
    return device.where(x > 0, x, alpha * x)
def _prelu(x, device, alpha):
    return device.where(x > 0, x, alpha * x)
def _elu(x, device, alpha):
    return device.where(x > 0, x, alpha * (device.exp(x) - 1))
def _gelu(x, device, alpha):
    return 0.5 * x * (1 + device.tanh(device.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))) 
def _swish(x, device, alpha):
    return x * (1 / (1 + device.exp(-x))) # More stable implementation: x * sigmoid(x)
def _selu(x, device, alpha):
    lam = 1.0507
    alpha_selu = 1.67326
    return lam * device.where(x > 0, x, alpha_selu * (device.exp(x) - 1))
def _softplus(x, device, alpha):
    return device.log(1 + device.exp(x))
def _mish(x, device, alpha):
    return x * device.tanh(device.log(1 + device.exp(x)))
def _rrelu(x, device, alpha):
    return device.where(x > 0, x, alpha * x)
def _hardswish(x, device, alpha):
    return x * device.where(x > 3, 1, device.where(x < -3, 0, (x + 3) / 6))
def _sigmoid(x, device, alpha):
    return 1 / (1 + device.exp(-x))
def _softsign(x, device, alpha):
    return x / (1 + device.abs(x))
def _tanh(x, device, alpha):
    return device.tanh(x)
def _hardtanh(x, device, alpha):
    return device.where(x > 1, 1, device.where(x < -1, -1, x))
def _hardsigmoid(x, device, alpha):
    return device.where(x > 1, 1, device.where(x < -1, 0, (x + 1) / 2))
def _tanhshrink(x, device, alpha):
    return x - device.tanh(x)
def _softshrink(x, device, alpha):
    return device.where(device.abs(x) > alpha, x - alpha * device.sign(x), 0)
def _hardshrink(x, device, alpha):
    return device.where(device.abs(x) > alpha, x, 0)
def _softmax(x, device, alpha):
    exp_x = device.exp(x - device.max(x, axis=-1, keepdims=True))
    return exp_x / device.sum(exp_x, axis=-1, keepdims=True)


# Activation Derivatives functions
def _relu_derivative(x, device, alpha):
    return device.where(x > 0, 1, 0)
def _leaky_relu_derivative(x, device, alpha):
    return device.where(x > 0, 1, alpha)
def _prelu_derivative(x, device, alpha):
    return device.where(x > 0, 1, alpha)
def _elu_derivative(x, device, alpha):
    return device.where(x > 0, 1, alpha * device.exp(x))
def _gelu_derivative(x, device, alpha):
    # This is a complex derivative, using a common approximation's derivative
    term1 = 0.5 * device.tanh(0.7978845608 * (x + 0.044715 * x**3))
    term2 = (0.3989422804 * x * (1 + 0.134145 * x**2)) / device.cosh(0.7978845608 * (x + 0.044715 * x**3))**2
    return 0.5 + term1 + term2
def _swish_derivative(x, device, alpha):
    sigmoid_x = 1 / (1 + device.exp(-x))
    return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)
def _selu_derivative(x, device, alpha):
    lam = 1.0507
    alpha_selu = 1.67326
    return lam * device.where(x > 0, 1, alpha_selu * device.exp(x))
def _softplus_derivative(x, device, alpha):
    return 1 / (1 + device.exp(-x))
def _mish_derivative(x, device, alpha):
    omega = device.exp(3*x) + 4*device.exp(2*x) + (6+4*x)*device.exp(x) + 4*(1+x)
    delta = 1 + 2*device.exp(x) + 2*device.exp(2*x) + device.exp(3*x)
    return (device.exp(x) * omega) / (delta**2)
def _rrelu_derivative(x, device, alpha):
    return device.where(x > 0, 1, alpha)
def _hardswish_derivative(x, device, alpha):
    return device.where(x > -3, device.where(x < 3, x / 3 + 0.5, 1), 0)
def _sigmoid_derivative(x, device, alpha):
    # Assumes x is the output of the sigmoid function, i.e., sigmoid(z)
    return x * (1 - x)
def _softsign_derivative(x, device, alpha):
    return 1 / (1 + device.abs(x)) ** 2
def _tanh_derivative(x, device, alpha):
    # Assumes x is the output of the tanh function, i.e., tanh(z)
    return 1 - x ** 2
def _hardtanh_derivative(x, device, alpha):
    return device.where(device.abs(x) <= 1, 1, 0)
def _hardsigmoid_derivative(x, device, alpha):
    return device.where(device.abs(x) <= 1, 0.5, 0)
def _tanhshrink_derivative(x, device, alpha):
    return device.tanh(x) ** 2
def _softshrink_derivative(x, device, alpha):
    return device.where(device.abs(x) > alpha, 1, 0)
def _hardshrink_derivative(x, device, alpha):
    return device.where(device.abs(x) > alpha, 1, 0)
def _softmax_derivative(x, device, alpha):
    # Assumes x is the output of the softmax function and used with categorical cross-entropy
    return x * (1 - x)


ACTIVATION_FUNCTIONS = {
    'relu': _relu,
    'leaky_relu': _leaky_relu,
    'prelu': _prelu,
    'elu': _elu,
    'gelu': _gelu,
    'swish': _swish,
    'selu': _selu,
    'softplus': _softplus,
    'mish': _mish,
    'rrelu': _rrelu,
    'hardswish': _hardswish,
    'sigmoid': _sigmoid,
    'softsign': _softsign,
    'tanh': _tanh,
    'hardtanh': _hardtanh,
    'hardsigmoid': _hardsigmoid,
    'tanhshrink': _tanhshrink,
    'softshrink': _softshrink,
    'hardshrink': _hardshrink,
    'softmax': _softmax,
}

ACTIVATION_DERIVATIVES = {
    'relu': _relu_derivative,
    'leaky_relu': _leaky_relu_derivative,
    'prelu': _prelu_derivative,
    'elu': _elu_derivative,
    'gelu': _gelu_derivative,
    'swish': _swish_derivative,
    'selu': _selu_derivative,
    'softplus': _softplus_derivative,
    'mish': _mish_derivative,
    'rrelu': _rrelu_derivative,
    'hardswish': _hardswish_derivative,
    'sigmoid': _sigmoid_derivative,
    'softsign': _softsign_derivative,
    'tanh': _tanh_derivative,
    'hardtanh': _hardtanh_derivative,
    'hardsigmoid': _hardsigmoid_derivative,
    'tanhshrink': _tanhshrink_derivative,
    'softshrink': _softshrink_derivative,
    'hardshrink': _hardshrink_derivative,
    'softmax': _softmax_derivative,
}

def activation(x, func, device, alpha=0.01):
    """
    Applies the specified activation function to the input data.

    Parameters:
    -----------
    x : np.ndarray or similar
        The input data to which the activation function will be applied.
    func : str
        The activation function to apply.
        Supported values: 'relu', 'leaky_relu', 'prelu', 'elu', 'gelu', 'swish', 'selu',
                          'softplus', 'mish', 'rrelu', 'hardswish', 'sigmoid', 'softsign',
                          'tanh', 'hardtanh', 'hardsigmoid', 'tanhshrink', 'softshrink',
                          'hardshrink', 'softmax'.
    device : object
        The computational device (CPU or GPU) that handles array operations.
    alpha : float, optional (default=0.01)
        Parameter used for activations like Leaky ReLU, PReLU, and RReLU.
    
    Returns:
    --------
    np.ndarray
        The result of applying the specified activation function to the input data.
    
    Raises:
    -------
    ValueError
        If the specified activation function is unsupported.
    """
    if func not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unsupported activation function: {func}")
   
    return ACTIVATION_FUNCTIONS[func](x, device, alpha)

def activation_derivative(x, func, device, alpha=0.01):
    """
    Computes the derivative of the specified activation function.

    Parameters:
    -----------
    x : np.ndarray or similar
        The input data on which the derivative will be computed.
    func : str
        The activation function whose derivative is to be computed.
        Supported values: 'relu', 'leaky_relu', 'prelu', 'elu', 'gelu', 'swish', 'selu',
                          'softplus', 'mish', 'rrelu', 'hardswish', 'sigmoid', 'softsign',
                          'tanh', 'hardtanh', 'hardsigmoid', 'tanhshrink', 'softshrink',
                          'hardshrink', 'softmax'.
    device : object
        The computational device (CPU or GPU) that handles array operations.
    alpha : float, optional (default=0.01)
        Parameter used for activations like Leaky ReLU, PReLU, and RReLU.
    
    Returns:
    --------
    np.ndarray
        The derivative of the activation function applied to the input data.
    
    Raises:
    -------
    ValueError
        If the specified activation function's derivative is unsupported.
    """
    if func not in ACTIVATION_DERIVATIVES:
        raise ValueError(f"Unsupported activation derivative: {func}")

    return ACTIVATION_DERIVATIVES[func](x, device, alpha)
