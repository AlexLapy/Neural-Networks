"""

Adding Layers to a Network -- Hidden Layers

We wish to add a hidden layer to our network. First, we need to make sure that the expected input to that layer
is the same as the output of the previous layer. The previous layer has three weight sets and three biases, so
we know it has three neurons. This then means, for the next layer, we can have as many weight sets as we want
( because this is how many neurons this new layer will have ), but each of those weigh sets must have three
discrete weights.

"""

import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights1 = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases1 = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

# The output of the first layer is a 3x1 matrix.
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1
print(layer1_outputs)

# The output of the second layer is a 3x1 matrix.
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)

import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init() # Sets random seed to 0, so we get the same random numbers each time we run the program.

# Create dataset
X, y = spiral_data(samples=100, classes=3)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()

""" 
Keep in mind that the neural network will nnot be aware of the classes. It will only see the data points.

Each dot is the feature, and its coordinates are the samples that form the dataset. The "classifications" for
that dot has to do with which spiral it is part of, depicted by the color of the dot. These colors would then
be assigned a class number for that model to fit to, like 0, 1, or 2.

"""

# Dense Layer Class ( Fully Connected Layer )


class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Let's see the output of the first few samples:
print(dense1.output[:5])

# The output is a 2D array, where each row is a sample, and each column is a neuron.

# Let's see the shape of the output:
print(dense1.output.shape)
