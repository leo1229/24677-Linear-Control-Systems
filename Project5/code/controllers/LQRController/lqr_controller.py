# LQR optimal controller

# Import libraries
import numpy as np
from base_controller import BaseController
from lqr_solver import dlqr, lqr
from scipy.linalg import solve_continuous_lyapunov, solve_lyapunov, solve_discrete_lyapunov
from math import cos, sin
import numpy as np
from scipy import signal

class LQRController(BaseController):
    """ The LQR controller class.

    """

    def __init__(self, robot, lossOfThurst=0):
        """ LQR controller __init__ method.

        Initialize parameters here.

        Args:
            robot (webots controller object): Controller for the drone.
            lossOfThrust (float): percent lost of thrust.

        """
        super().__init__(robot, lossOfThurst)

        # define integral error
        self.int_e1 = 0
        self.int_e2 = 0
        self.int_e3 = 0
        self.int_e4 = 0

        # define K matrix
        self.K = None

    def initializeGainMatrix(self):
        """ Calculate the gain matrix.

        """

        # ---------------|LQR Controller|-------------------------
        # Use the results of linearization to create a state-space model

        delT = self.timestep*1e-3
        g = self.g
        m = self.m
        Ix = self.Ix
        Iy = self.Iy
        Iz = self.Iz
        n_p = 12 # number of states
        m = 4 # number of integral error terms

        A = np.zeros((16, 16))
        A[0,6] = 1; A[1,7] = 1; A[2,8] = 1; A[3,9] = 1; A[4,10] = 1; A[5,11] = 1
        A[6,4] = g; A[7,3] = -g
        A[12,0] = 1; A[12,12] = -1
        A[13,1] = 1; A[13,13] = -1
        A[14,2] = 1; A[14,14] = -1
        A[15,5] = 1; A[15,15] = -1

        B = np.zeros((16, 4))
        B[8,0] = 1 / m; B[9,1] = 1 / Ix; B[10,2] = 1 / Iy; B[11,3] = 1 / Iz

        C = np.zeros((4,16))
        C[0,0] = 1; C[1,1] = 1; C[2,2] = 1; C[3,5] = 1

        D = np.zeros((4,4))

        sys_Con = signal.StateSpace(A,B,C,D)

        # ----------------- Your Code Here ----------------- #
        # Compute the discretized A_d, B_d, C_d, D_d, for the computation of LQR gain
        sys_Dis = sys_Con.to_discrete(delT)
        A_d = sys_Dis.A
        B_d = sys_Dis.B
        C_d = sys_Dis.C
        D_d = sys_Dis.D
        # ----------------- Your Code Ends Here ----------------- #



        # -----------------    Example code     ----------------- #
        # max_pos = 15.0
        # max_ang = 0.2 * self.pi
        # max_vel = 6.0
        # max_rate = 0.015 * self.pi
        # max_eyI = 3.

        # max_states = np.array([0.1 * max_pos, 0.1 * max_pos, max_pos,
        #                     max_ang, max_ang, max_ang,
        #                     0.5 * max_vel, 0.5 * max_vel, max_vel,
        #                     max_rate, max_rate, max_rate,
        #                     0.1 * max_eyI, 0.1 * max_eyI, 1 * max_eyI, 0.1 * max_eyI])

        # max_inputs = np.array([0.2 * self.U1_max, self.U1_max, self.U1_max, self.U1_max])

        # Q = np.diag(1/max_states**2)
        # R = np.diag(1/max_inputs**2)
        # -----------------  Example code Ends ----------------- #
        # ----------------- Your Code Here ----------------- #
        # Come up with reasonable values for Q and R (state and control weights)
        # The example code above is a good starting point, feel free to use them or write you own.
        # Tune them to get the better performance
        max_pos = 15.0
        max_ang = 0.2 * self.pi
        max_vel = 3.0
        max_rate = 0.015 * self.pi
        max_eyI = 0.9

        max_states = np.array([0.1 * max_pos, 0.1 * max_pos, max_pos,
                            max_ang, max_ang, max_ang,
                            0.5 * max_vel, 0.5 * max_vel, max_vel,
                            max_rate, max_rate, max_rate,
                            0.1 * max_eyI, 0.1 * max_eyI, 1 * max_eyI, 0.1 * max_eyI])

        max_inputs = np.array([0.2 * self.U1_max, self.U1_max, self.U1_max, self.U1_max])

        Q = np.diag(1/max_states**2)
        R = np.diag(1/max_inputs**2)

        # ----------------- Your Code Ends Here ----------------- #

        # solve for LQR gains
        [K, _, _] = dlqr(A_d, B_d, Q, R)

        self.K = -K

    def update(self, r):
        """ Get current states and calculate desired control input.

        Args:
            r (np.array): reference trajectory.

        Returns:
            np.array: states. information of the 16 states.
            np.array: U. desired control input.

        """

        # Fetch the states from the BaseController method
        x_t = super().getStates()

        # update integral term
        self.int_e1 += float((x_t[0]-r[0])*(self.timestep*1e-3))
        self.int_e2 += float((x_t[1]-r[1])*(self.timestep*1e-3))
        self.int_e3 += float((x_t[2]-r[2])*(self.timestep*1e-3))
        self.int_e4 += float((x_t[5]-r[3])*(self.timestep*1e-3))

        # Assemble error-based states into array
        error_state = np.array([self.int_e1, self.int_e2, self.int_e3, self.int_e4]).reshape((-1,1))
        states = np.concatenate((x_t, error_state))

        # calculate control input
        U = np.matmul(self.K, states)
        U[0] += self.g * self.m

        # Return all states and calculated control inputs U
        return states, U
