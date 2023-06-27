"""
The definining characteristic of a deep neural network is the presence of two or more hidden layers.
These hidden layers are ones that the neural network controls.
Most neural networks in use are a form of deep learning.

AI -> ML -> Neural Networks -> Deep Neural Networks


Superivsed Learning: The training data you feed to the algorithm includes the desired solutions, called labels.
Unsupervised Learning: The training data is unlabeled.

Each measure of the data is called a feature.
A group of features is called a feature set (vector or arrays).
The number of features is called the dimensionality of the dataset.
The value of a feature set can be referred to as a sample.

Samples are fed into the neural network model to train them to fit desired outputs from these inputs
or to predict based on them during the inference phase.

Samples are also called instances, observations, or records.

The "normal" and "failure" labels are classifications or labels. Also called targets or ground-truth labels.

For this book, we will focus on classification and regression with neural networks.


Dense layers, the most common layers, consist of interconnected neurons. In a dense layer, each neuron
of a given layer is connected to each neuron of the next layer, which mean that its output value becomes
an input for the next neurons.

Weights are the parameters that are learned by the neural network during training. They are the values  
that are adjusted by the training algorithm to fit the model to the data.

The bias is a constant value (+ or -) that is added to the input of a neuron. It is also a parameter
that is learned by the neural network during training.

Weights and biases can be thought of as the knobs that we can tune to fit our model to data.

Weight change the slope of the line, and bias changes the intercept of the line. (Output = Weight * Input + Bias)

The activation function is a function that is applied to the output of each neuron. It is usually a nonlinear
function that introduces nonlinearity to the model. The activation function is also called the transfer function.

Step function: 0 if x < 0, 1 if x >= 0, where x is the input to the neuron.

For a step function, if the neuron's output value, which is calculated by sum(weight * input) + bias, is
negative, the neuron's output is 0. If the output value is positive, the neuron's output is 1.

The formula for a single neuron is: output = activation(sum(weight * input) + bias)

While you can use a step function as an activation function, it is not a good choice for a neural network.
The reason is that the step function is not continuous, which means that a small change in the input can
cause a large change in the output. This is not good for a neural network because we want the neural network
to learn from small changes in the input.

A better choice for an activation function is the Rectified Linear Unit (ReLU) function. The ReLU function
is defined as: output = max(0, sum(weight * input) + bias)

The input data is typically preprocessed before it is fed into the neural network. This preprocessing
is done to make the data more suitable for the neural network. The preprocessing can include scaling
the data, normalizing the data, and need to be in numerical form.

Its common to preprocess the data while retaining its features and having the value in similar ranges
between 0 and 1 or -1 and 1. To achieve this, we can use either or both scaling and normalization.

The output layer is whatever the neural network returns. With classification, the output layer is
the probability of the input belonging to each class. The output layer, often has as many neurons
as the training dataset as classes, but we can also have a single output neuron for binary (two-class)
classification.

Overfitting is when the neural network model fits the training data too well. This means that the model
will not generalize well to new data. Overfitting can be caused by having too many parameters in the model
or by training the model for too long.

Generalization is the ability of the model to perform well on data that it has not seen before.

In-sample data is the data that the model has seen during training. Out-of-sample data is the data that
the model has not seen during training.

To train these neural networks, we calculate how "wrong" they are using algorithms to calculate the error
(called loss), and attempt to slowly reduce the error by adjusting the weights and biases.

They can be used for classification, regression, and clustering.
"""
