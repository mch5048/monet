import numpy as np

tmp = np.arange(10)

print(tmp)

l = []

for i in range(5):
    tmp = tmp * (i + 2)
    l.append(tmp)

print(l)
