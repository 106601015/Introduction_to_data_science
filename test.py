import numpy as np
import math

a = np.array(([0, 0, 0, 0, 0, 1, 1, 1, 2], [10, 10, 10, 10, 10, 11, 11, 11, 12]))
print(a.shape, a)

for i in a[:]:
    print(i)

x = np.array(a[:, 1]>15)
print('a', a)