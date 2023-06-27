""" 

Activation Functions

- Different activation functions are used for different purposes

- The activation function is a function that is applied to the output of each neuron. It is usually a nonlinear
function that introduces nonlinearity to the model. The activation function is also called the transfer function.

- STEP FUNCTION: 0 if x < 0, 1 if x >= 0, where x is the input to the neuron.

The purpose of the step function is to mimic a neuron that fires or does not fire. For a step function, if the
neuron's output value, which is calculated by sum(weight * input) + bias, is negative, the neuron's output is 0.
If the output value is positive, the neuron's output is 1.

The formula for a single neuron is: output = activation(sum(weight * input) + bias)

This activation function has been used in the past, but it is not used in modern neural networks. 


- LINEAR ACTIVATION FUNCTION: y = x

A straight line is a linear function.

This activation function is usually used in the output layer of a regression neural network. The reason is that
the output is a scalar value instead of a classificaiton.


- SIGMOID ACTIVATION FUNCTION: y = 1 / (1 + e^-x)

The sigmoid function is a nonlinear function that is used in the output layer of a binary classification neural
network. The reason is that the output is a probability value between 0 and 1, where the range of 0 to 0.5 is
one class and the range of 0.5 to 1 is the other class.

- THE RECTIFIED LINEAR UNIT (RELU) ACTIVATION FUNCTION: y = max(x, 0)

The ReLU function is defined as: output = max(sum(weight * input) + bias, 0)
It is very cloes to being a linear function, but it is not a linear function because it has a bend at x = 0.

In ReLU function, the bias offsets the line to the left or right, and the weight changes the slope of the line.
We're also able to control whether the function is one for determining where the neuron activates or deactivates.

"""

# ReLU Activation Function Code

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

output = np.maximum(0, inputs)
print(output)


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation func.
# Takes in output from previous layer
activation1.forward(dense1.output)

# Let's see the output of the first few samples:
print(activation1.output[:5])

""" 

Softmax Activation Function

- The softmax function is a nonlinear function that is used in the output layer of a multi-class classification
neural network. The reason is that the output is a probability value between 0 and 1, where the sum of all the
probabilities is 1.

- The softmax function is defined as: y = e^x / sum(e^x)

- The softmax function is a generalization of the sigmoid function. The sigmoid function is used in the output
layer of a binary classification neural network, and the softmax function is used in the output layer of a
multi-class classification neural network.

The exponential function is used to make the output positive. Also, the exponential function is a monotonic
function. This mean that, with higher input values, outputs are also higher, so we wont change the predicted class.
It also add stability to the result as the normalized exponentiation is more about the difference between numbers than
their magnitude. The sum of the exponential function of all the outputs is used to normalize the output to a probability
distribution.

"""

# For 1 sample

layers_outputs = [4.8, 1.21, 2.385]

# For each value in a vector, calculate the exponential value
exp_values = np.exp(layers_outputs)
print('exponentiated values:')
print(exp_values)

# Now normalize values
norm_values = exp_values / np.sum(exp_values)
print('normalized exponentiated values:')
print(norm_values)

# The sum of the normalized values is 1
print('sum of normalized values:', np.sum(norm_values))



# For multiple samples (batch)

layers_outputs = [[4.8, 1.21, 2.385],
                  [8.9, -1.81, 0.2],
                  [1.41, 1.051, 0.026]]

# For each value in a vector, calculate the exponential value
exp_values = np.exp(layers_outputs)
print('exponentiated values:')
print(exp_values)

# Sum each row
# Keepdims=True keeps the 2D structure instead of returning a 1D array
# axis=1 means sum across the rows
# axis=0 means sum across the columns
print(np.sum(layers_outputs, axis=1, keepdims=True))

# Now normalize values
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print('normalized exponentiated values:')
print(norm_values)

# The sum of the normalized values is 1
print('sum of normalized values:', np.sum(norm_values, axis=1, keepdims=True))


# Softmax Activation
class Activation_Softmax:
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
