from controller import GPS, Gyro, InertialUnit, Robot
import numpy as np

class BaseController():
    """ The base controller class.

    """

    def __init__(self, robot, lossOfThrust = 0):
        """ Base controller __init__ method.

        Initialize drone parameters here.

        Args:
            robot (webots controller object): Controller for the drone.
            lossOfThrust (float): percent lost of thrust.

        """
        # Initialize variables
        self.robot = robot
        self.timestep = 0

        # intializa percent loss of thrust
        self.lossOfThrust = lossOfThrust

        # Define robot parameter
        self.m = 0.4
        self.d1x = 0.1122
        self.d1y = 0.1515
        self.d2x = 0.11709
        self.d2y = 0.128
        self.Ix = 0.000913855
        self.Iy = 0.00236242
        self.Iz = 0.00279965

        # define constants
        self.g = 9.81
        self.ct = 0.00026
        self.ctau = 5.2e-06
        self.U1_max = 10
        self.pi = 3.1415926535

        # define H matrix for conversion from control input U to motor speeds
        self.H_inv = self.ct*np.array([[1, 1, 1, 1],
                                    [self.d1y, -self.d1y, self.d2y, -self.d2y],
                                    [-self.d1x, -self.d1x, self.d2x, self.d2x],
                                    [-self.ctau/self.ct, self.ctau/self.ct, self.ctau/self.ct, -self.ctau/self.ct]
                                    ])
        self.H = np.linalg.inv(self.H_inv)

        # define variables for speed calculations
        self.xGPS_old = 0
        self.yGPS_old = 0
        self.zGPS_old = 0.099019

    def startSensors(self, timestep):
        """ Start sensors.

        Instantiate objects and start up GPS, Gyro, IMU sensors.

        For more details, refer to the Webots documentation.

        Args: 
            timestep (int): time step of the current world.

        """
        self.gps = GPS("gps")
        self.gps.enable(timestep)

        self.gyro = Gyro("gyro")
        self.gyro.enable(timestep)

        self.imu = InertialUnit("inertial unit")
        self.imu.enable(timestep)

        self.timestep = timestep

    def getStates(self):
        """ Get drone state.

        The state of drone is 16 dimensional:

        xGPS, yGPS, zGPS, 
        roll, pitch, yaw, 
        x_vel, y_vel, z_vel,
        roll_rate, pitch_rate, yaw_rate

        Returns: 
            np.array: x_t. information of 12 states.

        """

        # Timestep returned by Webots is in ms, so we need to convert
        delT = 1e-3*self.timestep

        # Extract (X, Y, Z) coordinate from GPS
        xGPS = self.gps.getValues()[0]
        yGPS = -self.gps.getValues()[2]
        zGPS = self.gps.getValues()[1]

        # Find the rate of change in each axis, and store the current value of (X, Y, Z)
        # as previous (X, Y, Z) which will be used in the next call
        x_vel = (xGPS - self.xGPS_old)/delT
        y_vel = (yGPS - self.yGPS_old)/delT
        z_vel = (zGPS - self.zGPS_old)/delT

        self.xGPS_old = xGPS
        self.yGPS_old = yGPS
        self.zGPS_old = zGPS

        # Extract (roll, pitch, yaw) angle from imu
        roll = self.imu.getRollPitchYaw()[0] 
        pitch = -self.imu.getRollPitchYaw()[1]
        yaw = self.imu.getRollPitchYaw()[2]

        # Extract (roll rate, pitch rate, yaw rate) angular velocity from imu
        roll_rate = self.gyro.getValues()[0]
        pitch_rate = -self.gyro.getValues()[2] 
        yaw_rate = self.gyro.getValues()[1]

        x_t = np.array([xGPS, yGPS, zGPS, roll, pitch, yaw, x_vel, y_vel, z_vel, roll_rate, pitch_rate, yaw_rate]).reshape(-1,1)

        return x_t

    def getMotorAll(self):
        """ Get each motors' controller.

        Returns:
            list: Each motor's controller.

        """
        frontLeftMotor = self.robot.getMotor('front left propeller')
        frontRightMotor = self.robot.getMotor('front right propeller')
        backLeftMotor = self.robot.getMotor('rear left propeller')
        backRightMotor = self.robot.getMotor('rear right propeller')
        return [frontLeftMotor, frontRightMotor, backLeftMotor, backRightMotor]

    def initializeMotors(self):
        """ Initialisze all motors speed to 0.

        """
        [frontLeftMotor, frontRightMotor, backLeftMotor, backRightMotor] = self.getMotorAll()
        frontLeftMotor.setPosition(float('inf'))
        frontRightMotor.setPosition(float('inf'))
        backLeftMotor.setPosition(float('inf'))
        backRightMotor.setPosition(float('inf'))
        self.motorsSpeed(0, 0, 0, 0)

    def motorsSpeed(self, v1, v2, v3, v4):
        """ Set each motors' speed.

        Args:
            v1, v2, v3, v4 (int): desired speed for each motor.

        """
        [frontLeftMotor, frontRightMotor, backLeftMotor, backRightMotor] = self.getMotorAll()
        frontLeftMotor.setVelocity(v1)
        frontRightMotor.setVelocity(v2)
        backLeftMotor.setVelocity(v3)
        backRightMotor.setVelocity(v4)

    def convertUtoMotorSpeed(self, U):
        """ Convert control input to motor speed.

        Args:
            U (np.array): desired control input.

        Returns:
            np.array: rotorspeed. Desired rotor speed.

        """
        w_squre = np.clip(np.matmul(self.H, U), 0, 576**2)
        rotorspeed = np.sqrt(w_squre.flatten())
        return rotorspeed

    def setMotorsSpeed(self, motorspeed, motor_failure=0):
        """ Set motor speed.

        Args:
            motorspeed (np.array): desired motor speed.
            motor_failure (bool): True for motor failure, False otherwise.

        """
        if motor_failure:
            print("--- Motor Failure ---")
            factor = np.sqrt(1 - self.lossOfThrust)
            self.motorsSpeed(float(motorspeed[0]) * factor, float(-motorspeed[1]), float(-motorspeed[2]), float(motorspeed[3]))
        else:
            self.motorsSpeed(float(motorspeed[0]), float(-motorspeed[1]), float(-motorspeed[2]), float(motorspeed[3]))
