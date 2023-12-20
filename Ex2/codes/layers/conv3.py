import numpy as np

class Conv3:
    def __init__(self, in_channels, out_channels, filter_size, stride = 1, padding = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Weight = {'value': np.random.randn(out_channels, in_channels, filter_size, filter_size) * 1e-3,
                        'grad': np.zeros((out_channels, in_channels, filter_size, filter_size))}
        self.bias = {'value': np.zeros(out_channels),'grad': np.zeros(out_channels)}
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.mapping = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 0], [5, 0, 1],
                        [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 0], [4, 5, 0, 1],
                        [5, 0, 1, 2], [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5], [0, 1, 2, 3, 4, 5]]

    def forward(self, x):
        self.input = x
        N, C, H, W = x.shape
        padded_x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        self.Weight['value'] = np.random.randn(self.out_channels, self.in_channels, self.filter_size, self.filter_size) * 1e-3
        self.bias['value'] = np.zeros(self.out_channels)
        output_H = ((2 * self.padding - self.filter_size + H) // self.stride) + 1
        output_W = ((2 * self.padding - self.filter_size + W) // self.stride) + 1
        self.output = np.zeros((N, self.out_channels, output_H, output_W))

        for h_out in range(output_H):
            for w_out in range(output_W):
                h_start, h_end = h_out * self.stride, h_out * self.stride + self.filter_size
                w_start, w_end = w_out * self.stride, w_out * self.stride + self.filter_size
                receptive_field = padded_x[:, :, h_start:h_end, w_start:w_end]
                convolution_result = np.sum(receptive_field[:, np.newaxis, :, :, :] * self.Weight['value'],axis = (2, 3, 4))
                self.output[:, :, h_out, w_out] = convolution_result + self.bias['value']
        return self.output

    def backward(self, grad_back):
        N, C, H, W = self.input.shape
        padded_x = np.pad(self.input,((0, 0), (0, 0),(self.padding, self.padding),(self.padding, self.padding)))
        grad = np.zeros_like(padded_x)
        self.Weight['grad'] = np.zeros_like(self.Weight['value'])
        self.bias['grad'] = np.sum(grad_back, axis=(0, 2, 3))
        output_H = (2 * self.padding - self.filter_size + H) // self.stride + 1
        output_W = (2 * self.padding - self.filter_size + W) // self.stride + 1

        for h_out in range(output_H):
            for w_out in range(output_W):
                h_start, h_end = h_out * self.stride, h_out * self.stride + self.filter_size
                w_start, w_end = w_out * self.stride, w_out * self.stride + self.filter_size
                grad[:, :, h_start:h_end, w_start:w_end] += np.sum(grad_back[:, :, h_out, w_out].reshape((N, 1, 1, 1, -1)) *
                                                                    self.Weight['value'].transpose((1, 2, 3, 0)).reshape((1, self.in_channels,self.filter_size, self.filter_size, -1)),
                                                                    axis = 4)
                self.Weight['grad'] += np.sum(grad_back[:, :, h_out, w_out].T.reshape((-1, 1, 1, 1, N)) *
                                                padded_x[:, :, h_start:h_end, w_start:w_end].transpose((1, 2, 3, 0)),
                                                axis = 4)

        grad = grad[:, :, self.padding:self.padding + H, self.padding:self.padding + W]
        return grad