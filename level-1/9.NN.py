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

class Sigmoid():
    def forward(self, inp_layer):
        self.sigmoid = 1/(1+np.exp(inp_layer*-1))
        return self.sigmoid

    def backprop(self, dL):
        return  dL*self.sigmoid*(1-self.sigmoid)

def MSE(pred, target):
    batch = target.shape[0]
    L = 0.5* np.sum((target-pred)**2)/batch
    dL = -target + pred
    return  L, dL

def one_hot(input_list, size):
    output = np.zeros((len(input_list), size))
    for idx,_ in enumerate(input_list):
        output[idx, _] = 1
    return output

def flatten(input):
    return np.reshape(input, (input.shape[0],-1))

input_data = np.random.random((2,20))
target_data = input_data

# input data for mnist
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

one_hot_y_train = one_hot(y_train, 10)
flatten_x_train = flatten(x_train)

one_hot_y_test = one_hot(y_test, 10)
flatten_x_test = flatten(x_test)

dense1 = xW_b((784,30))
dense2  = xW_b((30,10))
sigmoid = Sigmoid()

for epoch in range(20):
    for _ in range(5000):
        input_data = flatten_x_train[_*10:_*10 + 10]
        y1 = dense1.forward(input_data)
        y2 = sigmoid.forward(y1)
        y3 = dense2.forward(y2)
        L, dL = MSE(y3, one_hot_y_train[_*10:_*10 + 10])

        #back propagation
        dL = dense2.backprop(dL, 0.001)
        dL = sigmoid.backprop(dL)
        dL = dense1.backprop(dL, 0.001)

        if _ % 1000 == 0:
            print("Loss",L)
    # evaluation 
    input_data = flatten_x_test
    y1 = dense1.forward(input_data)
    y2 = sigmoid.forward(y1)
    y3 = dense2.forward(y2)
    pred = np.argmax(y3, axis=1)
    L, dL = MSE(y3, one_hot_y_test)
    
    print("prediction : ",pred[:10])
    print("target     : ",y_test[:10])
    accuracy = 0
    for idx in range(len(pred)):
        if pred[idx] == y_test[idx]:
            accuracy += 1/len(pred)
    print("accuracy : {:.2f} %".format(100*accuracy))

