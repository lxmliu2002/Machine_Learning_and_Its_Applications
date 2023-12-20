from layers.conv import Conv
from layers.conv3 import Conv3
from layers.subsampling import MaxPooling
from layers.activate import SigMoid
from layers.linear import Linear



class LeNet5:
    """
    C1 -> S2 -> C3 -> S4 -> C5 -> F6 -> Output
    Reference: https://www.cnblogs.com/fengff/p/10173071.html
    """
    def __init__(self):
        self.C1 = Conv(1, 6, 5, 1, 0)
        self.s1 = SigMoid()
        self.S1 = MaxPooling((2, 2))
        self.s2 = SigMoid()
        self.C3 = Conv3(6, 16, 5, 1, 0)
        self.s3 = SigMoid()
        self.S4 = MaxPooling((2, 2))
        self.s4 = SigMoid()
        self.C5 = Linear(400, 120)
        self.s5 = SigMoid()
        self.F6 = Linear(120, 84)
        self.s6 = SigMoid()
        self.OUTPUT = Linear(84, 10)

    def forward(self, net):
        net = self.C1.forward(net)
        net = self.s1.forward(net)
        net = self.S1.forward(net)
        net = self.s2.forward(net)
        net = self.C3.forward(net)
        net = self.s3.forward(net)
        net = self.S4.forward(net)
        net = self.s4.forward(net)
        net = net.reshape(net.shape[0], -1)
        net = self.C5.forward(net)
        net = self.s5.forward(net)
        net = self.F6.forward(net)
        net = self.s6.forward(net)
        net = self.OUTPUT.forward(net)
        return net

    def backward(self, grad_back):
        grad_back = self.OUTPUT.backward(grad_back)
        grad_back = self.s6.backward(grad_back)
        grad_back = self.F6.backward(grad_back)
        grad_back = self.s5.backward(grad_back)
        grad_back = self.C5.backward(grad_back)
        grad_back = grad_back.reshape(grad_back.shape[0], 16, 5, 5)
        grad_back = self.s4.backward(grad_back)
        grad_back = self.S4.backward(grad_back)
        grad_back = self.s3.backward(grad_back)
        grad_back = self.C3.backward(grad_back)
        grad_back = self.s2.backward(grad_back)
        grad_back = self.S1.backward(grad_back)
        grad_back = self.s1.backward(grad_back)
        grad_back = self.C1.backward(grad_back)

    def get_params(self):
        return [self.C1.Weight, self.C1.bias, self.C3.Weight, self.C3.bias, self.C5.Weight, self.C5.bias, self.F6.Weight, self.F6.bias, self.OUTPUT.Weight, self.OUTPUT.bias]

    def set_params(self, params):
        self.C1.Weight = params[0]
        self.C1.bias = params[1]
        self.C3.Weight = params[2]
        self.C3.bias = params[3]
        self.C5.Weight = params[4]
        self.C5.bias = params[5]
        self.F6.Weight = params[6]
        self.F6.bias = params[7]
        self.OUTPUT.Weight = params[8]
        self.OUTPUT.bias = params[9]