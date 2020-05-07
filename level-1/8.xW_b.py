import numpy as np
import tensorflow as tf 

class xW_b:
    def __init__(self, hidden_unit):
        self.W = np.random.normal(size=[hidden_unit[0], hidden_unit[1]])*0.02
        self.b = np.random.normal(size=[hidden_unit[1]])*0.02

    def forward(self, x):
        self.x = x 
        self.batch = x.shape[0]
        output = np.matmul(self.x, self.W) + self.b
        return output

    def backprop(self, dL, lr):
        dL_prev = np.einsum('bo,io->bi', dL, self.W)
        db = np.sum(dL, axis=0)
        dW = np.einsum('bi,bo->io', self.x, dL)

        db = db*(1./self.batch)
        dW = dW*(1./self.batch)
        self.W = self.W - dW * lr
        self.b = self.b - db * lr
        return dL_prev

