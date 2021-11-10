https://page.mi.fu-berlin.de/rojas/neural/chapter/K3.pdf

#T(N,K) is the number of threshold functions a perceptron can remember given k input dimensionality, binary output, and n number of points

import sys
sys.setrecursionlimit(1500)

def tnk(m , n):
  if values[n][m] >= 0:
    return values[n][m]
  elif n == 0:
    values[n][m] = 0
    return 0
  elif m == 1 and n >=1:
    values[n][m] = 2
    return 2
  else:
    values[n][m] = R(m - 1, n) + R(m - 1, n - 1)
    return values[n][m]
