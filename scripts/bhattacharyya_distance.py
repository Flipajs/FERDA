from __future__ import print_function
from __future__ import unicode_literals
from builtins import range
import numpy as np
from math import log

def bhattacharyya(a, b):
    s = 0
    for i in range(len(a)):
        s += (a[i] * b[i]) ** 0.5

    print(a)
    print(b)
    print(s)
    print(-log(s))

a = [0.3, 0, 0., 0.1, 0.1, 0.5]
b = [0.3, 0, 0., 0.1, 0.1, 0.5]

bhattacharyya(a, b)
b = list(reversed([0.3, 0, 0., 0.1, 0.2, 0.5]))
bhattacharyya(a, b)

b = [0.0, 0.2, 0., 0.1, 0.2, 0.5]
bhattacharyya(a, b)

b = [0.0, 0.9, 0., 0.0, 0.1, 0]
bhattacharyya(a, b)