# coding=utf-8
import numpy as np


def predict(test_images, theta):
    scores = np.dot(test_images, theta.T)
    preds = np.argmax(scores, axis=1)
    return preds

def cal_accuracy(y_pred, y):
    # TODO: Compute the accuracy among the test set and store it in acc
    # check lenth
    if len(y_pred) != len(y):
        raise ValueError("The lengths of the predicted values and the actual labels are not consistent")

    correct = 0
    total = len(y)

    for i in range(total):
        if y_pred[i] == y[i]:
            correct += 1

    accuracy = (correct / total) * 100.0

    return accuracy