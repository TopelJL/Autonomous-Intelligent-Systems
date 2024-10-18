# ---------------------------------------------------------------------
#                               main.py
#                    University of Central Florida
#              Autonomous & Intelligent Systems Labratory
#
# Description: Central point of code to run from, sets up environment,
# vehicles, and calls multiple functions that ensure our sim runs 
# correct.
#
# Note(1): This research done was derived off of a paper by 
# Andrea Bajcsy, Somil Bansal, Eli Bronstein, Varun Tolani, and 
# Claire J. Tomlin.
#
# Note(2) Please see "An_Efficient_Reachability-Based_Framework_
# for_Provably_Safe_Autonomous_Navigation_in_Unknown_Environments"
# Inside of our documents folder for original work done.
# See website: https://doi.org/10.48550/arXiv.1905.00532
# See website: https://abajcsy.github.io/safe_navigation/
#
# Author        Date        Description
# ------        ----        ------------
# Jaxon Topel   9/27/24     Initial Matlab to python conversion
# Ryan Rahrooh   9/27/24     Iniital Matlab to python conversion
# ---------------------------------------------------------------------

# Functional Imports.
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Class imports.
import AvoidSet
import Plotter
import getAvoidingControlDubins

# ----------------
# CODE BEGINS HERE
# ----------------

# Clear the figure and variables
plt.clf()
plt.close('all')

# Setup environment bounds
low_env = np.array([0, 0])
up_env = np.array([10, 10])

# Create first obstacle
obs_shape = 'rectangle'  # Can be 'rectangle' or 'circle'

if obs_shape == 'rectangle':
    # Create lower and upper bounds on rectangle
    low_real_obs = np.array([3, 2])
    up_real_obs = np.array([6, 5])
elif obs_shape == 'circle':
    # Setup circular obstacle with center and radius
    low_real_obs = np.array([5.5, 2.5])  # Center of circle
    up_real_obs = np.array([1.5, 1.5])   # Radius of circle
else:
    raise ValueError('Unsupported obstacle shape')

# Setup computation domains and discretization
grid_low = np.concatenate([low_env, [-np.pi]])
grid_up = np.concatenate([up_env, [np.pi]])
N = np.array([31, 31, 21])  # Discretization steps

# Timestep for computation and simulation
dt = 0.05

# Initial and final locations for both vehicles
vehicle_1_init = np.array([2.0, 4.0, np.pi/2])
vehicle_2_init = np.array([9.0, 9.0, -np.pi/2])

goal_vehicle_1 = np.array([8, 1.5, -np.pi/2])
goal_vehicle_2 = np.array([9, 1.5, -np.pi/2])

# Construct sensed region
sense_shape = 'camera'  # Options: 'circle', 'camera'

if sense_shape == 'circle':
    sense_rad = 1.5
    sense_data_vehicle_1 = [vehicle_1_init, np.array([sense_rad, sense_rad])]
    sense_data_vehicle_2 = [vehicle_2_init, np.array([sense_rad, sense_rad])]
elif sense_shape == 'camera':
    initial_R = 1.5  # Initial radius of the safe region
    sense_FOV = np.pi / 6  # Field-of-view of the camera
    sense_rad = 1.5
    sense_data_vehicle_1 = [vehicle_1_init, np.array([sense_FOV, initial_R])]
    sense_data_vehicle_2 = [vehicle_2_init, np.array([sense_FOV, initial_R])]
else:
    raise ValueError("Unsupported sensor type")

# Update method and setup avoid sets
update_method = 'localQ'
warm_start = True
update_epsilon = 0.01
save_output_data = True
noise_mean = 0
noise_std = 0.5

# Create filename for saving results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"{update_method}_{'warm_' if warm_start else ''}{timestamp}.npz"

# Initialize AvoidSet objects for both vehicles
curr_time = 1

try:
    # Vehicle 1
    vehicle_1 = AvoidSet(grid_low, grid_up, low_real_obs, up_real_obs, obs_shape, vehicle_1_init, N, dt, update_epsilon, warm_start, update_method, noise_mean, noise_std, sense_rad)
    vehicle_1.compute_avoid_set(sense_data_vehicle_1, sense_shape, curr_time, vehicle_1_init)

    # Vehicle 2
    vehicle_2 = AvoidSet(grid_low, grid_up, low_real_obs, up_real_obs, obs_shape, vehicle_2_init, N, dt, update_epsilon, warm_start, update_method, noise_mean, noise_std, sense_rad)
    vehicle_2.compute_avoid_set(sense_data_vehicle_2, sense_shape, curr_time, vehicle_2_init)

except Exception as e:
    print(f"Error initializing AvoidSet objects: {e}")
    raise

# Plot initial conditions
fig, ax = plt.subplots()
plt.ion()  # Turn on interactive mode
plt.show()

# Plot environment, car, and sensing regions
plotter = Plotter(low_env, up_env, low_real_obs, up_real_obs, obs_shape)
plotter.update_plot(vehicle_1_init, vehicle_1)
plotter.update_plot(vehicle_2_init, vehicle_2)
plt.draw()

# Simulate Dubins car movement
T = 200  # Number of time steps
for t in range(T):
    try:
        # Get current control for both vehicles
        u1 = getAvoidingControlDubins(t, vehicle_1.dyn_sys.x, goal_vehicle_1, vehicle_1)
        u2 = getAvoidingControlDubins(t, vehicle_2.dyn_sys.x, goal_vehicle_2, vehicle_2)

        # Apply control to dynamics
        vehicle_1.dyn_sys.update_state(u1, dt, vehicle_1.dyn_sys.x)
        vehicle_2.dyn_sys.update_state(u2, dt, vehicle_2.dyn_sys.x)

        # Update positions
        x1 = vehicle_1.dyn_sys.x
        x2 = vehicle_2.dyn_sys.x

        # Add Gaussian noise to sensing
        vehicle_1.add_gaussian_noise(x1)
        vehicle_2.add_gaussian_noise(x2)

        # Update sensed data
        if sense_shape == 'circle':
            sense_data_vehicle_1 = [x1, np.array([sense_rad, sense_rad])]
            sense_data_vehicle_2 = [x2, np.array([sense_rad, sense_rad])]
        elif sense_shape == 'camera':
            sense_data_vehicle_1 = [x1, np.array([sense_FOV, initial_R])]
            sense_data_vehicle_2 = [x2, np.array([sense_FOV, initial_R])]

        # Update obstacle info based on sensed data
        vehicle_1.update_obstacle_detection(x1, low_real_obs, up_real_obs)
        vehicle_2.update_obstacle_detection(x2, low_real_obs, up_real_obs)

        # Compute avoid set for both vehicles
        vehicle_1.compute_avoid_set(sense_data_vehicle_1, sense_shape, t + 1, x1)
        vehicle_2.compute_avoid_set(sense_data_vehicle_2, sense_shape, t + 1, x2)

        # Update plot
        plotter.update_plot(x1, vehicle_1)
        plotter.update_plot(x2, vehicle_2)
        plt.draw()
        plt.pause(0.001)  # Use plt.pause instead of time.sleep for better plot responsiveness

        # Log progress
        print(f"Time Step: {t + 1} completed")

    except Exception as e:
        print(f"Error during simulation at time step {t + 1}: {e}")
        break

# Save output data
if save_output_data:
    try:
        np.savez(filename, valueFunCellArr=vehicle_1.value_fun_cell_arr, lxCellArr=vehicle_1.lx_cell_arr, QSizeCellArr=vehicle_1.q_size_cell_arr,
                 solnTimes=vehicle_1.soln_times, fovCellArr=vehicle_1.fov_cell_arr)
        print(f"Simulation data saved to {filename}")
    except Exception as e:
        print(f"Error saving output data: {e}")
