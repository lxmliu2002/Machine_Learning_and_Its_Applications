# coding=utf-8
import numpy as np


def predict(test_images, theta):
    scores = np.dot(test_images, theta.T)
    preds = np.argmax(scores, axis=1)
    return preds

# def cal_accuracy(y_pred, y):
#     # TODO: Compute the accuracy among the test set and store it in acc
    
#     return acc
def cal_accuracy(y_pred, y):
    # TODO: Compute the accuracy among the test set and store it in acc

    # 将y展平成1维向量
    y = y.flatten()
    # 正确个数
    right = np.sum(y_pred == y)
    # 总个数
    total = y_pred.shape[0]
    acc = right/total

    return acc