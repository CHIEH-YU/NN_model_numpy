import numpy as np


myinput = 10*(np.random.random((2,5)) - 0.5)


def sigmoid(_input):
    return 1/(1+np.exp(-1*_input))
print(myinput)
print(sigmoid(myinput))
