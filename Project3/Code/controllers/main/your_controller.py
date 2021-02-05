# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81

        # Add additional member variables according to your need here.
        self.velocity_previous_error = 0
        self.velocity_cumulative_error = 0

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        _, closest_point = closestNode(X, Y, trajectory)

        ahead = 20

        if (closest_point + ahead >= trajectory.shape[0]):
            ahead = 0

        # Get the desired values from the closest_point.
        V_desired = 20
        X_should = trajectory[closest_point + ahead, 0]
        Y_should = trajectory[closest_point + ahead, 1]
        psi_should = np.arctan2(Y_should - Y, X_should - X)

        # Design your controllers in the spaces below.
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta).

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        """
        Vx = xdot
        A = np.array([[0, 1, 0, 0],
        [0, -4*Ca / (m * Vx), 4*Ca/m, -(2*Ca*(lf - lr))/(m*Vx)],
        [0, 0, 0, 1],
        [0, -(2*Ca*(lf - lr)) / (Iz * Vx), (2*Ca*(lf - lr)) / Iz, (-2*Ca*(lf*lf + lr * lr)) / (Iz * Vx)]])

        B = np.array([[0], [2*Ca / m], [0], [(2 * Ca* lf) / Iz]])
        C = np.identity(4)
        D = np.array([[0],[0],[0],[0]])

        # Discrete the system.
        sys_Con = signal.StateSpace(A,B,C,D)
        sys_Dis = sys_Con.to_discrete(delT)
        A = sys_Dis.A
        B = sys_Dis.B

        e1 = (np.power(np.power(X_should - X, 2) + np.power(Y_should - Y, 2), 0.5))
        e2 = wrapToPi(psi - psi_should)
        e1_dot = ydot + xdot * e2
        e2_dot = psidot

        e = np.hstack((e1, e1_dot, e2, e2_dot)).reshape(4,1)

        q1 = 0.01
        q2 = 1
        q3 = 0.01
        q4 = 1

        Q = np.array([[q1, 0, 0, 0],
                      [0, q2, 0, 0],
                      [0, 0, q3, 0],
                      [0, 0, 0, q4]])

        R = 25

        S = np.matrix(linalg.solve_discrete_are(A, B, Q, R))
        K = -np.matrix(linalg.inv(B.T @ S @ B + R) * (B.T @ S @ A))

        delta = wrapToPi(np.dot(K,e)[0,0])
        if (delta < -3.1416 / 6):
            delta = -3.1415 / 6
        elif(delta > 3.1416 / 6):
            delta = 3.1416 / 6


        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        """
        velocity_error = V_desired - xdot
        velocity_differential_error = (velocity_error - self.velocity_previous_error) / delT
        self.velocity_cumulative_error += velocity_error * delT
        self.velocity_previous_error = velocity_error

        kp2 = 50
        ki2 = 0.0001
        kd2 = 0.0001
        F = kp2 * velocity_error + ki2 * self.velocity_cumulative_error + kd2 * velocity_differential_error
        if(F < 0):
            F = 0
        elif(F > 15736):
            F = 15736


        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
