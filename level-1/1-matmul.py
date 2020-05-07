import numpy as np


a = np.random.random((4))
b = np.random.random((4,2))


c = np.matmul(a,b)
# what is the shape of c?
print(c)



a = np.random.random((5,4))
b = np.random.random((4,2))


c = np.matmul(a,b)
# what is the shape of c?
print(c)


a = np.random.random((2,5,4))
b = np.random.random((4,2))


c = np.matmul(a,b)
# what is the shape of c?
print(c.shape)



