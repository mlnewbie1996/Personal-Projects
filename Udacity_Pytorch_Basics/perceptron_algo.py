import numpy as np 

np.random.seed(42)

def stepfunction(t):
    if t>=0:
        return 1
    return 0

def prediction(X,W,b):
    return stepfunction((np.matmul(X,W)+b)[0])

def perceptronStep(X,y,W,b,learn_rate=0.01):
    for i in range(len(X)):
        x1 = X[i][0]
        x2 = X[i][1]
        # This is the function that defines the perceptron algorithm
        label = 1 if x1 * W[0] + x2 * W[1] + b > 0 else 0.0

        if label > y[i]:
            W[0] = W[0] - x1 * learn_rate
            W[1] = W[1] - x2 * learn_rate
            b = b - learn_rate
        elif label < y[i]:
            W[0] = W[0] + x1 * learn_rate
            W[1] = W[1] + x2 * learn_rate
            b = b + learn_rate
    return W, b

def trainPreceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[1])
    y_min, y_max = max(y.T[0]), max(y.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines