# This is the code I use for HW5 exercise5 question d.
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')

x1 = np.linspace(-10, 10, 1000)
x2 = np.linspace(-100, 100, 1000)
x1Square = np.multiply(x1, x1)
x1Biquadrate = np.multiply(x1Square, x1Square)
x2Square = np.multiply(x2, x2)
z = -4 * np.multiply(x1Biquadrate, x2Square)

ax.plot(x1, x2, z)
plt.title("the variation of V derivatie with respect to x1 and x2")
ax.set_xlabel('x label', color='r')
ax.set_ylabel('y label', color='g')
ax.set_zlabel('V derivatie label', color='b')
plt.show()
