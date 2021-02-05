# This file should be set as the controller for the DJI Maverick node.
# Please do not alter this file - it may cause the simulation to fail.

# Import Webots libraries
from controller import Robot

import numpy as np

# Import evalution functions
from eval import evaluateLQR

# Import functions from other scripts in controller folder
from lqr_controller import LQRController
# from adaptive_controller import CustomController

# Instantiate dron driver supervisor
driver = Robot()

# Get the time step of the current world
timestep = int(driver.getBasicTimeStep())

# Instantiate controller and start sensors
customController = LQRController(driver)
customController.initializeMotors()
customController.startSensors(timestep)

# Initialize state storage vectors
stateHistory = []
referenceHistory = []

# calculate gain matrix for baseline LQR controller & adaptive controller
customController.initializeGainMatrix()

# start simulation
while driver.step(timestep) != -1:

    current_time = driver.getTime()
    print("Time:", current_time)

    # reference trajectory
    if current_time < 10:
        r = np.array([0, 0, 2, 0]).reshape(-1,1)
    elif current_time >= 10 and current_time < 20:
        r = np.array([0, 0, 3, 0]).reshape(-1,1)
    elif current_time >= 20 and current_time < 30:
        r = np.array([0, 0, 2, 0]).reshape(-1,1)
    elif current_time >= 30 and current_time < 40:
        r = np.array([0, 0, 4, 0]).reshape(-1,1)
    elif current_time >= 40 and current_time < 50:
        r = np.array([0, 0, 1, 0]).reshape(-1,1)
    elif current_time >= 50 and current_time < 60:
        r = np.array([0, 0, 4, 0]).reshape(-1,1)
    else:
        # end simulation
        break

    # Call control update method
    states, U = customController.update(r)

    # Check failure
    if (states[2] < 0 ):
        print("="*15 + "Drone Crashed" + "="*15)
        print("="*15 + "Your Controller Failed" + "="*15)

    # Convert control input to motoespeed
    rotorspeed = customController.convertUtoMotorSpeed(U)
    print(rotorspeed)

    # set motor speed 
    customController.setMotorsSpeed(rotorspeed)

    # collect state history for evaluation
    stateHistory.append(list(states.flatten()))
    referenceHistory.append(list(r.flatten()))

# save data for evaluation
# reference trajectory
np.save("r_hist_ex1", referenceHistory)
# states using lqr controller
np.save("x_hist_ex1", stateHistory)

# simulation finished, calculate score
evaluateLQR()