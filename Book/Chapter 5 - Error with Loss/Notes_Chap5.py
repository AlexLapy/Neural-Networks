"""

Calculating Network Error with Loss

The loss function, also refered to as the cost function, is the algorithm that quantifies how wrong a model is.
Loss is the measure of this metric. Since loss is the model's error, the lower the loss, the better the model.

We can't use argmax() to calculate the loss because argmax() is not a continuous function. We need a continuous
function to calculate the gradient. We need a function that is continuous and has a minimum value at the correct
prediction. We need a function that is 0 when the prediction is correct and increases as the prediction gets worse.

Categorical Cross-Entropy Loss

If you are familiar with regression, you know one of the loss function used with neural network that do regression:
Squared error (or mean squared error with neural networks).

For classifiying data, we use a different loss function: Categorical cross-entropy loss.This one is explicitly used
to compare "ground-truth" probability (y or "target") and some predicted distribution (y_hat or "prediction"). Is
is one of the most commonly used loss functions with a softmax activation function on the output layer.

The formula for calculating categorical cross-entropy of y (actual/desired distribution) and y_hat (predicted
distribution) is:

L_i = -sum_j(y_i_j * log(y_hat_i_j))

Where L_i denotes sample loss value, i is the i-th sample in the set, j is the j-th label/output index, y_i_j denotes
the target values, and y_hat_i_j denotes the predicted values.

We can simplify this formula by using the fact that the target values are all zeros except for the index of the
correct class. We can also use the fact that the log of 1 is 0. This simplifies the formula to:

L_i = -log(y_hat_i_k) , where k is the index of the correct class

"""

import math
import numpy as np

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)

# We can simplify this formula by using the fact that the target values are all zeros except for the index of the
# correct class. We can also use the fact that the log of 1 is 0. This simplifies the formula to:

k = target_output.index(1)
loss = -math.log(softmax_output[k])

print(loss)

""" 

For example: with confidence level of [0.22, 0.6, 0.18] or [0.32, 0.36, 0.32]. In both case, the argmax of these
will return the second class as the prediction, but the model's confidence about these predictions is high only
for one of them. The Catetorical cross-entropy loss function account for that and outputs a larger loss the lower
the confidence of the model is about the prediction.

"""
print("Example of loss output")
print(math.log(1))
print(math.log(0.9))
print(math.log(0.8))
print(math.log(0.7))
print(math.log(0.6))
print(math.log(0.5))
print(math.log(0.4))
print(math.log(0.3))
print(math.log(0.2))
print(math.log(0.1))
print(math.log(0.01))


"""

Categorical Cross-Entropy Loss with NumPy

Example: there are 3 samples in the dataset, each with 3 classes.
"dog" is class 0, "cat" is class 1, and "human" is class 2.

"""
softmax_output = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])
class_targets = [0, 1, 1]

# Index a numpy array with a list of indexes
output_rows_list = range(len(softmax_output))
confidence_at_targets_index = softmax_output[output_rows_list, class_targets]
print(confidence_at_targets_index)

# Calculate sample losses
neg_log = -np.log(confidence_at_targets_index)
print(neg_log)

# Calculate average loss
print(np.mean(neg_log))

"""

Target can be one-hot encoded or sparse. If the target is one-hot encoded, we can use the same formula as before.
If the target is sparse, we can use the same formula as before, but we need to index the softmax output with the
target index instead of the target vector.

"""

softmax_output = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])
class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])

# Probabilities for target values -
# only if categorical labels
if len(class_targets.shape) == 1:
    correct_confidences = softmax_output[range(len(softmax_output)), class_targets]

# Mask values - only for one-hot encoded labels
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(softmax_output * class_targets, axis=1)

# Losses
neg_log = -np.log(correct_confidences)
average_loss = np.mean(neg_log)
print(average_loss)

"""

Cliping Values

We can clip values to avoid division by zero errors. We can clip values to avoid log(0) errors. We can clip values
to avoid negative losses.

"""

softmax_output = np.array([[0.7, 0.1, 0.2],
                           [0, 1, 0],
                           [0.02, 0.9, 0.08]])

softmax_output_clipped = np.clip(softmax_output, 1e-7, 1 - 1e-7)
print(softmax_output_clipped)

"""

Categorical Cross-Entropy Loss Class

"""


class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss
    

class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


"""

Example

"""

softmax_output = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])
class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(softmax_output, class_targets)
print(loss)


"""

Accuracy Calculation

Describe how often the largest confidence is the correct class in term of a fraction. 
We can use argmax() on the softmax output and then compare it to the target vector. 

"""

# Probabilities of 3 samples
softmax_output = np.array([[0.7, 0.1, 0.2],
                           [0.5, 0.1, 0.4],
                           [0.02, 0.9, 0.08]])
# Target (ground-truth) labels for 3 samples
class_targets = np.array([0, 1, 1])

# Calculate values along second axis (axis of index 1)
predictions = np.argmax(softmax_output, axis=1)

# If targets are one-hot encoded - convert them to sparse
if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1)

# True evaluates to 1; False to 0
accuracy = np.mean(predictions == class_targets)
print(accuracy)
