import numpy as np

# dot
a = np.random.random((3))
b = np.random.random((3))


c = np.dot(a,b)
print(c.shape)



# matmul
a = np.random.random((4,3))
b = np.random.random((3,5))


c = np.dot(a,b)
print(c.shape)
