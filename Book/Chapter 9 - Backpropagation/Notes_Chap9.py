""" 

Backpropagation

The derivative with respect to the weights and a bias will inform us about their impact and will be used to update them.
The derivative with respect to the inputs will inform us about the impact of the inputs on the output of the previous layer.
or
The derivative with respect to the inputs are used to chain more layers by passing them to the previous function in the chain.

Example: We have a list of 3 samples for input, where each sample has 4 features.
         We have a single hidden layer with 3 neurons (list of 3 sets of weights and a bias).
"""

import numpy as np

# ** THE GRADIENT WITH RESPECT TO THE INPUTS **

# Passed-in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# Sum weights related to the given input multiplied by
# the gradient related to the given neuron
dinputs = np.dot(dvalues, weights.T)

print(dinputs)


# ** THE GRADIENT WITH RESPECT TO THE WEIGHTS **

dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

dweights = np.dot(inputs.T, dvalues)

print(dweights)


# ** THE GRADIENT WITH RESPECT TO THE BIASES **

dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

biases = np.array([[2, 3, 0.5]])

# sum values, do this over samples (first axis), keepdims
dbiases = np.sum(dvalues, axis=0, keepdims=True)

print(dbiases)


# ** THE GRADIENT FOR RELU **

z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])

dvalues = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

drelu = np.zeros_like(z)
drelu[z > 0] = 1

print(drelu)

drelu *= dvalues

print(drelu)


# SIMPLIFIED VERSION

drelu = dvalues.copy()
drelu[z <= 0] = 0

print(drelu)


"""

Example of minimazing the Relu's output

"""

dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

biases = np.array([[2, 3, 0.5]])

# Forward pass
layer_outputs = np.dot(inputs, weights) + biases
relu_outputs = np.maximum(0, layer_outputs)

# Let's optimize and test backpropagation here
# ReLU activation - simulates derivative with respect to input values
# from next layer passed to current layer during backpropagation
drelu = dvalues.copy()
drelu[layer_outputs <= 0] = 0

# Dense layer
dinputs = np.dot(drelu, weights.T)
dweights = np.dot(inputs.T, drelu)
dbiases = np.sum(drelu, axis=0, keepdims=True)

# Update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights)
print(biases)


"""

** Categorical Cross-Entropy Loss derivative **

dinputs = -y_true / y_pred

dinputs = dinputs / samples, for normalization

We normalize the gradient by dividing it by the number of samples,
so that the learning rate is not dependent on the batch size.


** Softmax activation derivative **

jacobian = softmax_outputs * np.eye(sofmax_outputs.shape[0]) - np.dot(softmax_outputs, softmax_outputs.T)
or
jacobian = np.diagflat(softmax_output) - np.dot(softmax_outputs, softmax_outputs.T)

dinputs[index] = np.dot(jacobian, single_dvalues)


** Combined Softmax activation and Categorical Cross-Entropy Loss derivative **

dinputs = y_pred - y_true    (substraction of the predicted and ground truth values)
dinputs = dvalues.copy()
dinputs[range(samples), y_true] -= 1

dinputs = dinputs / samples  (normalization)

"""


