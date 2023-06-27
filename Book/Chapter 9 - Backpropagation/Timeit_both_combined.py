import Neural_Network_Class as nn
from timeit import timeit
import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([0, 1, 1])


def f1():
    softmax_loss = nn.Activation_Softmax_Loss_CategoricalCrossentropy()
    softmax_loss.backward(softmax_outputs, class_targets)
    dvalues1 = softmax_loss.dinputs
    print(dvalues1)


def f2():
    activation = nn.Activation_Softmax()
    activation.output = softmax_outputs
    loss = nn.Loss_CategoricalCrossentropy()
    loss.backward(softmax_outputs, class_targets)
    activation.backward(loss.dinputs)
    dvalues2 = activation.dinputs
    print(dvalues2)


t1 = timeit(lambda: f1(), number=1)
t2 = timeit(lambda: f2(), number=1)

print(t2/t1)
