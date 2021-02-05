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

        # Add
        self.psi_cumulative_error = 0
        self.velocity_cumulative_error = 0
        self.psi_previous_error = 0
        self.velocity_previous_error = 0


        # Add additional member variables according to your need here.

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

        # Find the closest point to the car in the trajectory.
        _, closest_point = closestNode(X, Y, trajectory)

        ahead = 15

        if (closest_point + ahead >= trajectory.shape[0]):
            ahead = 0

        # Get the desired values from the closest_point.
        X_should = trajectory[closest_point + ahead, 0]
        Y_should = trajectory[closest_point + ahead, 1]
        psi_should = np.arctan2(Y_should - Y, X_should - X)

        # Design your controllers in the spaces below.
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta).


        # ---------------|Lateral Controller|-------------------------
        psi_error = wrapToPi(psi_should - psi)
        psi_differential_error = (psi_error - self.psi_previous_error) / delT
        self.psi_cumulative_error += psi_error * delT
        self.psi_previous_error = psi_error

        kp1 = 3.5
        ki1= 0.001
        kd1 = 0.001

        delta = kp1 * psi_error + ki1 * self.psi_cumulative_error + kd1 * psi_differential_error


        if (delta < -3.1416 / 6):
            delta = -3.1415 / 6
        elif(delta > 3.1416 / 6):
            delta = 3.1416 / 6



        # ---------------|Longitudinal Controller|-------------------------
        velocity_error = (np.power(np.power(X_should - X, 2) + np.power(Y_should - Y, 2), 0.5)) / delT
        velocity_differential_error = (velocity_error - self.velocity_previous_error) / delT
        self.velocity_cumulative_error += velocity_error * delT
        self.velocity_previous_error = velocity_error

        kp2 = 10
        ki2 = 0.0001
        kd2 = 0.0001
        F = kp2 * velocity_error + ki2 * self.velocity_cumulative_error + kd2 * velocity_differential_error
        if (F > 15736):
            F = 15736
        elif(F < 1000):
            F = 1000

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
