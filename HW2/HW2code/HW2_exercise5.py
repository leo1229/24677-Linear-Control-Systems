# This is the code for Exercise 5.
import numpy as np
from scipy.signal import StateSpace, dlsim
A = np.asarray([[0.,1.],
                [1.,1.]])
B = np.asarray([[0.],[0.]])
C = np.asarray([0.,0.])
D = np.asarray([0.])
plane_sys = StateSpace(A, B, C, D, dt = 1)
t = np.arange(0, 20, 1)
input = np.zeros(len(t))
_, y, x = dlsim(plane_sys, input, t, x0=[0, 1])
F20 = int(x[19, 1])
print("F20 is:", F20)
