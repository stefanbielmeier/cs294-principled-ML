
#T(N,K) is the number of threshold functions a perceptron can remember given k input dimensionality, binary output, and n number of points
#Capacity of the neuron without the bias: 2 bits per weight

import sys
sys.setrecursionlimit(1500)

import scipy.special


#k denotes number of inputs to the neuron (+ bias), n the number of data points
n = 10000
k = 3073

values = []

#how many functions can perceptron memorize for k = dimensionality of data (+ 1 bias), and n = number of data points given (which determines
#number of possible boolean functions
def tnk(n, k):
  if k >= n:
    return 2**n
  else:
    value = 0
    for i in range(k):
      value += scipy.special.comb(n-1, i, exact=True)
    return value*2

print(tnk(n,k)/2**n)