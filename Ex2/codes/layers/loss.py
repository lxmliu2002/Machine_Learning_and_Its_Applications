import numpy as np

class SoftMax:

    # def softmax_loss(y_pred, y):
    #     # y_pred: (N, C) - 预测的类别概率分布
    #     # y: (N, 1) - 实际的类别标签

    #     N = y_pred.shape[0]  # 样本数量
    #     ex = np.exp(y_pred)  # 计算预测的类别概率的指数
    #     sumx = np.sum(ex, axis = 1, keepdims = True)  # 保持维度一致性

    #     # 计算 softmax 损失
    #     loss = np.mean(-np.log(ex[range(N), y] / sumx.flatten()))

    #     # 计算 softmax 梯度
    #     grad = ex / sumx
    #     grad[range(N), y] -= 1
    #     grad /= N

    #     # 计算分类准确率
    #     acc = np.mean(np.argmax(ex, axis=1) == y.flatten())

    #     return loss, grad, acc

    def loss(pred, y):
        probabilities = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
        correct_class_probabilities = probabilities[range(pred.shape[0]), y]
        loss = -np.mean(np.log(correct_class_probabilities))
        return loss

    def gradient(pred, y):
        num_samples = pred.shape[0]
        probabilities = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
        probabilities[range(num_samples), y] -= 1
        gradient = probabilities / num_samples
        return gradient

    def accuracy(pred, y):
        predicted_classes = np.argmax(pred, axis=1)
        accuracy = np.mean(predicted_classes == y.flatten())
        return accuracy

    # def loss(y_pred, y):
    #     # loss = np.mean(np.log(np.sum(np.exp(y_pred), axis = 1, keepdims = True)) - y_pred[range(y_pred.shape[0]), list(y)])
    #     return np.mean(np.log(np.sum(np.exp(y_pred), axis = 1, keepdims = True)) - y_pred[range(y_pred.shape[0]), list(y)])

    # def gradient(y_pred, y):
    #     num = y_pred.shape[0]  # 样本数量
    #     grad = y_pred / np.sum(np.exp(y_pred), axis = 1, keepdims = True)
    #     grad[range(num), y] = grad[range(num), y] - 1
    #     grad = grad / num
    #     return grad

    # def accuracy(y_pred, y):
    #     # acc = np.mean(np.argmax(y_pred, axis = 1) == y.flatten())
    #     return np.mean(np.argmax(y_pred, axis = 1) == y.flatten())



# def softmax_loss(y_pred, y):
#     # y_pred: (N, C) - 预测的类别概率分布
#     # y: (N, 1) - 实际的类别标签

#     N = y_pred.shape[0]  # 样本数量
#     ex = np.exp(y_pred)  # 计算预测的类别概率的指数
#     sumx = np.sum(ex, axis = 1, keepdims = True)  # 保持维度一致性

#     # 计算 softmax 损失
#     loss = calculate_loss(y_pred, y, sumx)

#     # 计算 softmax 梯度
#     grad = calculate_gradient(y_pred, y, sumx, N)

#     # 计算分类准确率
#     acc = calculate_accuracy(y_pred, y)

#     return loss, grad, acc


    def cross_entropy_loss(y_pred, y):
        # y_pred: (N, C)
        # y: (N, 1)
        N = y_pred.shape[0]

        # 使用交叉熵损失计算
        loss = -np.mean(np.log(y_pred[range(N), list(y)]))

        # 计算梯度
        grad = np.zeros_like(y_pred)
        grad[range(N), list(y)] -= 1
        grad /= N

        # 计算准确率
        acc = np.mean(np.argmax(y_pred, axis=1) == y.reshape(1, y.shape[0]))

        return loss, grad, acc

