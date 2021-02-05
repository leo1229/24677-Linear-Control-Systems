import numpy as np
from scipy.signal import StateSpace, dlsim
import matplotlib.pyplot as plt
#define some constant value
G = np.asarray([[1.0, 0.2, 0.1],
                [0.1, 2.0, 0.1],
                [0.3, 0.1, 3.0]])
r = 3
a = 1.2
e = 0.1
# Build the system
A = np.asarray([[0., a*r*G[0][1]/G[0][0], a*r*G[0][2]/G[0][0]],
                [a*r*G[1][0]/G[1][1], 0., a*r*G[1][2]/G[1][1]],
                [a*r*G[2][0]/G[2][2], a*r*G[2][1]/G[2][2], 0.]])
B = np.asarray([[a*r/G[0][0]],
                [a*r/G[1][1]],
                [a*r/G[2][2]]])
C = np.asarray([0., 0., 0.])
D = np.asarray([0.])
plane_sys = StateSpace(A, B, C, D, dt = 1)
#define the simulation time step
t = np.arange(0, 30, 1)
#Simulate the system to get p
input = e * e * np.ones(len(t))
_, y, x = dlsim(plane_sys, input, t, x0=[0.1, 0.01, 0.02])
p = x.T
#Calculate S
def calculateS(p1, p2, p3, G1, G2, G3):
    s = np.zeros(len(p1))
    for i in range(len(p1)):
        s[i] = G1 * p1[i] / (e*e + G2*p2[i] + G3*p3[i])
    return s
S1 = calculateS(p[0], p[1], p[2], G[0][0], G[0][1], G[0][2])
S2 = calculateS(p[1], p[0], p[2], G[1][1], G[1][0], G[1][2])
S3 = calculateS(p[2], p[0], p[1], G[2][2], G[2][0], G[2][1])
S = np.asarray([S1, S2, S3])

#plot figure of p
plt.figure(1)
for i in range(3):
    plt.plot(t, p[i], label = 'p' + str(i+1))
plt.ylabel('power level')
plt.xlabel('t [s]')
plt.legend()
plt.show()
#plot figure of S
plt.figure(2)
for i in range(3):
    plt.plot(t, S[i], label = 'S' + str(i+1))
plt.plot(t, a*r*np.ones(len(t)), label = 'Desired SINR')
plt.ylabel('ratio')
plt.xlabel('t [s]')
plt.legend()
plt.show()
