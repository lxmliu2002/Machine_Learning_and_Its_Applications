import numpy as np


# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    c = np.max(x, axis = 1, keepdims = True)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x, axis = 1, keepdims = True)
    return exp_x / sum_exp_x


# 损失函数
# 均方损失
def MSE(y, t):
    return np.sum((y - t) ** 2)


# 交叉熵损失
def CEE(y, t):
    d = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + d)) / batch_size


# 激活层
"""
relu:在正向传播时将所有负数的输入转换为0，
而在反向传播时将所有负数的输入位置的导数转换为0。
"""


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


"""
sigmoid:在正向传播时将所有输入转换为0到1之间的值，
而在反向传播时将所有输入位置的导数转换为0到1之间的值。
"""


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


# 全连接层
class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None
        self.original_x_shape = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.w) + self.b
        return out

    def backward(self, dout):
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        dx = np.dot(dout, self.w.T)
        return dx.reshape(*self.original_x_shape)


# 分类头
class SoftmaxWithCrossEntropy:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = CEE(self.y, self.t)
        return self.loss

    def backward(self, dout = 1):
        batch_size = self.y.shape[0]
        return (self.y - self.t) / batch_size


# 正则化优化器
class Dropout:
    def __init__(self, ratio = 0.2):
        self.ratio = ratio
        self.mask = None

    def forward(self, x, train = True):
        if train:
            self.mask = np.random.rand(x.shape[0], x.shape[1]) > self.ratio
            return x * self.mask
        else:
            return x * (1 - self.ratio)

    def backward(self, dout):
        return dout * self.mask


# 归一化 https://blog.csdn.net/qq_42118719/article/details/113544673
class BatchNormalization:
    def __init__(self, gamma, beta, momentum = 0.9, running_mean = None, running_var = None):
        # gamma: 缩放因子
        # beta: 平移因子
        # momentum: 滑动平均的参数
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        # 滑动平均+滑动方差
        self.running_mean = running_mean
        self.running_var = running_var

        # 反向传播时使用
        self.batch_size = None
        self.x_ = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg = True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        # 训练模式要更新滑动平均和滑动方差
        if train_flg:
            sample_mean = x.mean(axis = 0)
            sample_var = x.var(axis = 0)
            self.batch_size = x.shape[0]
            self.var_plus_eps = sample_var + 10e-7
            self.x_ = (x - sample_mean) / np.sqrt(sample_var + 10e-7)

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * sample_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * sample_var
            )
        else:
            # 测试模式使用滑动平均和滑动方差
            xc = x - self.running_mean
            self.x_ = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * self.x_ + self.beta
        return out.reshape(*self.input_shape)

    def backward(self, dout):
        # 梯度计算
        self.dgamma = np.sum(self.x_ * dout, axis = 0)
        self.dbeta = np.sum(dout, axis = 0)

        # 反向传播，这里是链式法则，先计算dx_，再计算dx，最后做一个归一化
        dx_ = (
            np.matmul(np.ones((self.batch_size, 1)), self.gamma.reshape((1, -1))) * dout
        )
        dx = (
            self.batch_size * dx_
            - np.sum(dx_, axis = 0)
            - self.x_ * np.sum(dx_ * self.x_, axis=0)
        )
        dx *= (1.0 / self.batch_size) / np.sqrt(self.var_plus_eps)

        return dx


"""
https://blog.csdn.net/m0_38065572/article/details/104709433
im2col:将输入数据展开以适用于卷积运算的函数
col2im:将展开的数据还原成输入数据的函数
"""


# image to column------------------------------------------------------
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride = 1, pad = 0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(
        0, 3, 4, 5, 1, 2
    )

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad : H + pad, pad : W + pad]


# 卷积层
class Convolution:
    def __init__(self, W, b, stride = 1, pad = 0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中间数据（backward 时使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.xshape = x.shape
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.xshape, FH, FW, self.stride, self.pad)

        return dx


class MaxPooling:
    def __init__(self, pool_h, pool_w, stride = 2, pad = 0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis = 1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.xshape = x.shape
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.xshape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


# 梯度下降，这部分原理见pdf
class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v == None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:

    """AdaGrad"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop:

    """RMSprop"""

    def __init__(self, lr = 0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:

    """Adam"""

    def __init__(self, lr = 0.001, beta1 = 0.9, beta2 = 0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = (
            self.lr
            * np.sqrt(1.0 - self.beta2**self.iter)
            / (1.0 - self.beta1**self.iter)
        )

        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (
                grads[key] ** 2
            )
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
