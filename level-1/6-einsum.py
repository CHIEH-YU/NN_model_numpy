import numpy as np


# this is magic

# sum, dot, matmul, ... and some other variances are possible.



a = np.random.random((3,4))
b = np.random.random((4,5))


c = np.matmul(a,b)

d = np.einsum('ij,jk->ik',a,b)
print(d)

#############################

a = np.random.random((2,3,4))
b = np.random.random((2,4,3))


c = np.matmul(a,b)

d = np.einsum('ijk,ikd->ijd',a,b)


# sum
e = np.einsum('ijk->i',a)



