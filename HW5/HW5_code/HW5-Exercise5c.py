# This is the code I use for HW5 Exercise5 question 5
import numpy as np
import matplotlib.pyplot as plt

def f_original(X):
    x1, x2 = X
    return [x2 - x1 * x2 * x2, -x1 * x1 * x1]

def f_linearized(X):
    x1, x2 = X
    return [x2, 0]

x1_original = np.linspace(-5, 5, 20)
x2_original = np.linspace(-10, 10, 20)

x1_linearized = np.linspace(-0.2, 0.2, 20)
x2_linearized = np.linspace(-0.2, 0.2, 20)

X1_original, X2_original = np.meshgrid(x1_original, x2_original)
X1_linearized, X2_linearized = np.meshgrid(x1_linearized, x2_linearized)

u_original, v_original = np.zeros(X1_original.shape), np.zeros(X2_original.shape)
u_linearized, v_linearized = np.zeros(X1_linearized.shape), np.zeros(X2_linearized.shape)

NI_original, NJ_original = X1_original.shape
NI_linearized, NJ_linearized = X1_linearized.shape

for i in range(NI_original):
    for j in range(NJ_original):
        x1 = X1_original[i, j]
        x2 = X2_original[i, j]
        y_original = f_original([x1, x2])
        u_original[i,j] = y_original[0]
        v_original[i,j] = y_original[1]

for i in range(NI_linearized):
    for j in range( NJ_linearized):
        x1 = X1_original[i, j]
        x2 = X2_original[i, j]
        y_linearized = f_linearized([x1, x2])
        u_linearized[i,j] = y_linearized[0]
        v_linearized[i,j] = y_linearized[1]

plt.figure()
Q1 = plt.quiver(X1_original, X2_original, u_original, v_original, color='r')
plt.title("The Phase Portrait plot of the original system")
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()


plt.figure()
Q2 = plt.quiver(X1_linearized, X2_linearized, u_linearized, v_linearized, color='b')
plt.title("The Phase Portrait plot of the linearized system")
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()
