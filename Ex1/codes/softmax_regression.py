# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt


def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and
    # the objective function value of every iteration and update the theta
    
    # print("theta.shape",theta.shape) (10, 784)       theta[10, 784]
    # print("y.shape",y.shape) (10, 60000)             y[10, 60000]
    # print("x.shape",x.shape) (60000, 784)            x[60000, 784]
    m = x.shape[0]
    # print(m) m = 60000                               m = 60000
    loss_history = []
    for i in range(iters):
    # for i in tqdm(range(iters), desc="Training"):
        scores = np.dot(theta, x.T)  # Calculate scores for each class
        # print(scores.shape) (10, 60000)
        exp_scores = np.exp(scores - np.max(scores, axis = 0))  # Subtract max score for numerical stability
        # print(exp_scores.shape) (10, 60000)
        probabilities = exp_scores / np.sum(exp_scores, axis = 0)  # Calculate class probabilities
        # print("probabilities",probabilities.shape) (10, 60000)
        
        # probabilities = np.exp(np.matmul(theta, x.T)) / np.sum(np.exp(np.matmul(theta, x.T)), axis=0)
        f = -np.sum(y * np.log(probabilities)) / m
        loss_history.append(f)
        # Calculate the gradient
        g = np.dot((probabilities - y), x) / m
        # Update theta using gradient descent
        theta -= alpha * g
        
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{iters}, f: {f}")


    plt.plot(range(1, iters + 1), loss_history)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iterations')
    plt.show()

    return theta