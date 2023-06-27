import math
import numpy as np

def create_venn_data(samples, classes):
    total_samples = samples * classes
    X = np.zeros((total_samples, 2))
    y = np.zeros((total_samples, classes), dtype='uint8')
    radius = 0.4

    if classes == 2:
        centers = [[-0.25, 0.25], [0.25, -0.25]]
    elif classes == 3:
        centers = [[-0.25, 0.25], [0, -0.25], [0.25, 0.25]]
    elif classes == 4:
        centers = [[-0.25, 0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, -0.25]]

    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        center = centers[class_number]
        center_x = center[0]
        center_y = center[1]
        r = np.linspace(0.0, radius, samples)
        t = np.linspace(class_number * 20, (class_number + 1)
                        * 20, samples) + np.random.randn(samples) * 0.2
        X[ix] = np.c_[center_x + r *
                      np.sin(t * 2.5), center_y + r * np.cos(t * 2.5)]

    for i in range(total_samples):
        xi = X[i, 0]
        yi = X[i, 1]
        for class_number, center in enumerate(centers):
            center_x = center[0]
            center_y = center[1]
            distance = math.sqrt((xi - center_x)**2 + (yi - center_y)**2)
            if distance <= radius+0.02:
                y[i, class_number] = 1

    return X, y
