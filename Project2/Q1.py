import numpy as np
import matplotlib.pyplot as plt
import control

# define the value of all the constant variables.
Ca = 20000
m = 1888.6
lf = 1.55
lr = 1.39
Iz = 25854

# Check the controllability and observability of the system at the following
# longitudinal velocities: 2 m/s, 5 m/s and 8 m/s.
for i in range(3):
    if i == 0:
        Vx = 2
    elif i == 1:
        Vx = 5
    else:
         Vx = 8
    A = np.array([[0, 1, 0, 0],
    [0, -4*Ca / (m * Vx), 4*Ca/m, -(2*Ca*(lf - lr))/(m*Vx)],
    [0, 0, 0, 1],
    [0, -(2*Ca*(lf - lr)) / (Iz * Vx), (2*Ca*(lf - lr)) / Iz, (-2*Ca*(lf*lf + lr * lr)) / (Iz * Vx)]])

    B = np.array([[0, 0], [2*Ca / m, 0], [0, 0], [(2 * Ca* lf) / Iz, 0]])

    C = np.identity(4)

    P = np.hstack((B, np.dot(A, B), np.dot(np.linalg.matrix_power(A, 2), B), np.dot(np.linalg.matrix_power(A, 3), B)))
    Q = np.vstack((C, np.dot(C, A), np.dot(C, np.linalg.matrix_power(A, 2)), np.dot(C, np.linalg.matrix_power(A, 3))))
    rankP = np.linalg.matrix_rank(P)
    rankQ = np.linalg.matrix_rank(Q)
    print("When Vx = ", Vx, "m/s:")
    if rankP == 4:
        print("The rank of P is", rankP, ",so this system is controllable.")
    else:
        print("The rank of P is", rankP, ",so this system is not controllable.")
    if rankQ == 4:
        print("The rank of Q is", rankQ, ",so this system is observable.")
    else:
        print("The rank of Q is", rankQ, ",so this system is not observable.")
    print()

# Exercise 1 question 2.
velocity = np.linspace(1, 40, 1000).reshape(1000,1)
rate = np.empty([1000,1])
poles = np.empty([1000,4])
for i in range(velocity.shape[0]):
    Vx = velocity[i]

    # Build the system.
    A = np.array([[0, 1, 0, 0],
    [0, -4*Ca / (m * Vx), 4*Ca/m, -(2*Ca*(lf - lr))/(m*Vx)],
    [0, 0, 0, 1],
    [0, -(2*Ca*(lf - lr)) / (Iz * Vx), (2*Ca*(lf - lr)) / Iz, (-2*Ca*(lf*lf + lr * lr)) / (Iz * Vx)]], dtype = float)
    B = np.array([[0, 0], [2*Ca / m, 0], [0, 0], [(2 * Ca* lf) / Iz, 0]])
    C = np.identity(4)
    D = np.array([[0, 0],[0, 0],[0, 0],[0, 0]])
    sys = control.StateSpace(A, B, C, D)

    # Calculate the value of logarithm
    P = np.hstack((B, np.dot(A, B), np.dot(np.linalg.matrix_power(A, 2), B), np.dot(np.linalg.matrix_power(A, 3), B)))
    _, delt,_ = np.linalg.svd(P)
    delt1 = max(delt)
    deltn = min(delt)
    rate[i] = np.log10(delt1 / deltn)

    #Calculate the poles of the system.
    poles_this_time = control.pole(sys)
    for j in range(4):
        poles[i,j] = poles_this_time[j].real


# Plot the logarithm of the greatest singular value divided by the smallest versus Vx (m/s)
plt.figure(1)
plt.title("The logarithm of the greatest singular value divided by the smallest versus Vx (m/s)")
plt.plot(velocity, rate)
plt.xlabel("Vx m/s")
plt.ylabel("log10(delt1 / deltn)")
plt.show()

# Plot the real part of the poles versus Vx (m/s).
plt.figure(2)

plt.subplot(2, 2, 1)
plt.xlabel("Vx m/s")
plt.ylabel("Re(p1)")
plt.plot(velocity, poles[:,0])

plt.subplot(2, 2, 2)
plt.xlabel("Vx m/s")
plt.ylabel("Re(p2)")
plt.plot(velocity, poles[:,1])

plt.subplot(2, 2, 3)
plt.xlabel("Vx m/s")
plt.ylabel("Re(p3)")
plt.plot(velocity, poles[:,2])

plt.subplot(2, 2, 4)
plt.xlabel("Vx m/s")
plt.ylabel("Re(p4)")
plt.plot(velocity, poles[:,3])

plt.tight_layout()
plt.show()
