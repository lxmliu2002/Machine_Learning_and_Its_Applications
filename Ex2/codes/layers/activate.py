import numpy as np

class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad_back):
        return grad_back * (self.x > 0)


class SigMoid:
    def forward(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))

    def backward(self, grad_back):
        sigmoid_x = 1 / (1 + np.exp(-self.x))
        return grad_back * sigmoid_x * (1 - sigmoid_x)