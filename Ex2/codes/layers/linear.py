import numpy as np

class Linear:
    def __init__(self, input_size, output_size):
        # super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.Weight = {'value': np.random.randn(input_size, output_size) * 1e-3, 'grad': np.zeros((input_size, output_size))}
        self.bias = {'value': np.zeros(output_size), 'grad': np.zeros(output_size)}

    def forward(self, x):
        self.input = x
        self.Weight['value'] = np.random.randn(self.input_size, self.output_size) * 1e-3
        self.bias['value'] = np.zeros(self.output_size)
        # self.output = np.dot(x, self.Weight['value']) + self.bias['value']
        self.output = np.add(np.dot(x, self.Weight['value']), self.bias['value'])
        return self.output


    def backward(self, grad_back):
        # Compute the gradient with respect to the input
        grad = np.dot(grad_back, self.Weight['value'].T)

        # Initialize gradient arrays
        self.Weight['grad'] = np.zeros((self.input_size, self.output_size))
        self.bias['grad'] = np.zeros(self.output_size)

        # Compute the gradient with respect to weights and biases
        self.Weight['grad'] = np.dot(self.input.T, grad_back)
        self.bias['grad'] = np.sum(grad_back, axis=0)

        return grad