"""
Let's say we have a single neuron, and there are three inputs to the neuron.
As in most cases, when you initialize parameters in neural networks, you initialize
the weights randomly, and you initialize the biases to zero.

The inputs will be either actual training data or the outputs of neurons form the previous layer.

"""

import numpy as np

# Single neuron with three inputs
inputs = np.array([1, 2, 3])
weights = np.array([0.2, 0.8, -0.5])
bias = 2

output = np.dot(weights, inputs) + bias
print(output)

# Now let's say we have three neurons in a layer, and each neuron has four inputs.
# We can represent the weights as a 3x4 matrix, and the biases as a 3x1 matrix.
# The inputs will be a 4x1 matrix.
inputs = np.array([1, 2, 3, 2.5])
weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])
biases = np.array([2, 3, 0.5])

# The output of the layer is a 3x1 matrix.
output = np.dot(weights, inputs) + biases
print(output)

"""

Tensors, Arrays, and Vectors

A tensor is a generalization of vectors and matrices. A vector is a 1D tensor, and a matrix is a 2D tensor.
A tensor can have any number of dimensions.

A tensor is a container for data. The data in a tensor is of the same type, such as float32, int32, or uint8.
The data is stored in a contiguous block of memory.

A tensor has a shape, which is the number of dimensions it has and the size of each dimension.
For example, a 2x3 matrix has a shape of (2, 3), and a 3D tensor with a shape of (3, 3, 5) has three dimensions
with each dimension having a size of 3, 3, and 5, respectively.

Homologous dimensions are dimensions that have the same size. For example, a 2x3 matrix has two homologous
dimensions, and a 3D tensor with a shape of (3, 3, 5) has two homologous dimensions, which are the first
and second dimensions.

Matrices are 2D tensors, and vectors are 1D tensors. Arrays can any number of dimensions, and they are
called tensors in TensorFlow.

A tensor object is an object that can be represented as an array.

In this book, we define an array as an ordered homogenous container for numbers.

Dot product is the sum of the products of the corresponding entries of the two sequences of numbers.

Vector addition is adding the corresponding entries of the two vectors. 


A batch of data is a set of data samples. A batch of data is fed into the neural network at a time.
The batch size is the number of samples in a batch of data.

The batch size is a hyperparameter that is set before training the neural network. The batch size is
usually set to a power of 2, such as 32, 64, 128, and so on.

To train, neural networks tend to receive data in batches. So far we have been feeding the neural network
one sample (or observation) at a time.

Two reasons for using batches of data:
1. The first reason is that it is more efficient to train the neural network using batches of data.
    By utilizing parallel processing, the training time can be reduced.
2. The second reason is that it is more accurate to train the neural network using batches of data.
    The reason is that the gradient of the loss function is calculated using the average loss of the
    batch of data. This is more accurate than using the loss of a single sample.

Matrix multiplication is a linear operation that takes two matrices as inputs and produces a matrix as output.
The number of columns of the first matrix must be equal to the number of rows of the second matrix.

Row and column vectors are special cases of matrices. A row vector is a matrix with a single row, and a
column vector is a matrix with a single column.

The dot product of two vectors is a scalar. The matrix multiplication of a row vector and a column vector is a matrix of size 1x1.
We perform matrix multiplication instead of dot product when we use a row vector and a column vector.

Transposition for the Matrix Product

The dot product of two vectors equals a matrix product of a row vector and a column vector.


"""

inputs = np.array([[1, 2, 3, 2.5],
                   [2.0, 5.0, -1.0, 2.0],
                   [-1.5, 2.7, 3.3, -0.8]])
weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])
biases = np.array([2, 3, 0.5])

# The output of the layer is a 3x1 matrix.
layer_outputs = np.dot(inputs, weights.T) + biases
print(layer_outputs)
