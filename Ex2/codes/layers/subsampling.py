import numpy as np

class MaxPooling:
    def __init__(self, kernel_size = (2, 2)):
        # super().__init__()
        self.kernel_size = kernel_size
        # self.input = None
        # self.output = None

    # def forward(self, x):
    #     kernel_size_h, kernel_size_w = self.kernel_size
    #     N, C, H, W = x.shape
    #     self.input = x.copy()
    #     self.output = np.zeros((N, C, H // kernel_size_h, W // kernel_size_w))
    #     for h in range(H // kernel_size_h):
    #         for w in range(W // kernel_size_w):
    #             self.output[:, :, h, w] = np.max(x[:, :, h*kernel_size_h:(h+1)*kernel_size_h, w*kernel_size_w:(w+1)*kernel_size_w], axis = (2, 3))
    #     return self.output
    def forward(self, x):
        kernel_size_h, kernel_size_w = self.kernel_size
        # N, C, H, W = x.shape
        self.input = x

        # Reshape input to prepare for max pooling
        reshaped_x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2] // kernel_size_h, kernel_size_h, x.shape[3] // kernel_size_w, kernel_size_w))

        # Apply max pooling
        self.output = np.max(reshaped_x, axis=(3, 5))

        return self.output



    # def backward(self, back_grad):
    #     kernel_size_h, kernel_size_w = self.kernel_size
    #     N, C, H, W = self.input.shape
    #     grad = np.zeros_like(self.input)
    #     for h in range(H // kernel_size_h):
    #         for w in range(W // kernel_size_w):
    #             tmp_x = self.input[:, :, h*kernel_size_h:(h+1)*kernel_size_h, w*kernel_size_w:(w+1)*kernel_size_w].reshape((N, C, -1))
    #             mask = np.zeros((N, C, kernel_size_h*kernel_size_w))
    #             mask[np.arange(N)[:, None], np.arange(C)[None, :], np.argmax(tmp_x, axis = 2)] = 1
    #             grad[:, :, h*kernel_size_h:(h+1)*kernel_size_h, w*kernel_size_w:(w+1)*kernel_size_w] = mask.reshape((N, C, kernel_size_h, kernel_size_w)) * back_grad[:, :, h, w][:, :, None, None]
    #     return grad

    def backward(self, grad_back):
        kernel_size_h, kernel_size_w = self.kernel_size
        grad = np.zeros_like(self.input)

        for h in range(0, self.input.shape[2], kernel_size_h):
            for w in range(0, self.input.shape[3], kernel_size_w):
                input_slice = self.input[:, :, h:h + kernel_size_h, w:w + kernel_size_w]
                input_slice_reshaped = input_slice.reshape((self.input.shape[0], self.input.shape[1], -1))

                argmax_indices = np.argmax(input_slice_reshaped, axis=2)

                mask = np.eye(kernel_size_h * kernel_size_w, dtype=bool)[argmax_indices].reshape(
                    (self.input.shape[0], self.input.shape[1], kernel_size_h, kernel_size_w))

                # 使用花式索引更新grad
                idx_h, idx_w = np.indices((kernel_size_h, kernel_size_w))
                grad[:, :, h:h + kernel_size_h, w:w + kernel_size_w] = mask[:, :, idx_h, idx_w] * grad_back[:, :, h // kernel_size_h, w // kernel_size_w][:, :, None, None]

        return grad