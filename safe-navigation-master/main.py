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
# Ryan Rahroo   9/27/24     Iniital Matlab to python conversion
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
obs_shape = 'rectangle'

if obs_shape == 'rectangle':
    # Create lower and upper bounds on rectangle
    low_real_obs = np.array([3, 2])
    up_real_obs = np.array([6, 5])
else:
    # Setup circular obstacle
    low_real_obs = np.array([5.5, 2.5])  # Center of circle
    up_real_obs = np.array([1.5, 1.5])   # Radius of circle

# Setup computation domains and discretization
grid_low = np.concatenate([low_env, [-np.pi]])
grid_up = np.concatenate([up_env, [np.pi]])
N = np.array([31, 31, 21])

# Timestep for computation and simulation
dt = 0.05

# Initial and final locations
x_init = np.array([2.0, 4.0, np.pi/2])
y_init = np.array([9.0, 9.0, -np.pi/2])

x_goal = np.array([8, 1.5, -np.pi/2])
y_goal = np.array([9, 1.5, -np.pi/2])

# Construct sensed region
sense_shape = 'camera'
if sense_shape == 'circle':
    sense_rad = 1.5
    sense_data1 = [x_init, np.array([sense_rad, sense_rad])]
    sense_data2 = [y_init, np.array([sense_rad, sense_rad])]
elif sense_shape == 'camera':
    initial_R = 1.5  # Initial radius of the safe region
    sense_FOV = np.pi / 6  # Field-of-view of the camera
    sense_rad = 1.5
    sense_data1 = [x_init, np.array([sense_FOV, initial_R])]
    sense_data2 = [y_init, np.array([sense_FOV, initial_R])]
else:
    raise ValueError("Unknown sensor type")

# Update method and setup avoid sets
update_method = 'localQ'
warm_start = True
update_epsilon = 0.01
save_output_data = True
noise_mean = 0
noise_std = 0.5

if warm_start:
    filename = f"{update_method}_warm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
else:
    filename = f"{update_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"

# First vehicle AvoidSet object
curr_time = 1
obj1 = AvoidSet(grid_low, grid_up, low_real_obs, up_real_obs, obs_shape, x_init, N, dt, update_epsilon, warm_start, update_method, noise_mean, noise_std, sense_rad)
obj1.compute_avoid_set(sense_data1, sense_shape, curr_time, x_init)

# Second vehicle AvoidSet object
obj2 = AvoidSet(grid_low, grid_up, low_real_obs, up_real_obs, obs_shape, y_init, N, dt, update_epsilon, warm_start, update_method, noise_mean, noise_std, sense_rad)
obj2.compute_avoid_set(sense_data2, sense_shape, curr_time, y_init)

# Plot initial conditions
fig, ax = plt.subplots()
plt.ion()  # Turn on interactive mode
plt.show()

# Plot environment, car, and sensing regions
plt_obj = Plotter(low_env, up_env, low_real_obs, up_real_obs, obs_shape)
plt_obj.update_plot(x_init, obj1)
plt_obj.update_plot(y_init, obj2)
plt.draw()
time.sleep(dt)

# Simulate Dubins car movement
T = 200
x = obj1.dyn_sys.x
y = obj2.dyn_sys.x

for t in range(T):
    # Get current control
    u1 = getAvoidingControlDubins(t, obj1.dyn_sys.x, x_goal, obj1)
    u2 = getAvoidingControlDubins(t, obj2.dyn_sys.x, y_goal, obj2)

    # Apply control to dynamics
    obj1.dyn_sys.update_state(u1, dt, obj1.dyn_sys.x)
    obj2.dyn_sys.update_state(u2, dt, obj2.dyn_sys.x)

    x = obj1.dyn_sys.x
    y = obj2.dyn_sys.x

    # Compute new sensing regions with noise
    if sense_shape == 'circle':
        print(f'(x, y, heading) position of 1st vehicle without noise: {x}')
        obj1.add_gaussian_noise(x)

        print(f'(x, y, heading) position of 2nd vehicle without noise: {y}')
        obj2.add_gaussian_noise(y)

        sense_data1 = [x, np.array([sense_rad, sense_rad])]
        sense_data2 = [y, np.array([sense_rad, sense_rad])]
    elif sense_shape == 'camera':
        print(f'(x, y, heading) position of 1st vehicle without noise: {x}')
        obj1.add_gaussian_noise(x)

        print(f'(x, y, heading) position of 2nd vehicle without noise: {y}')
        obj2.add_gaussian_noise(y)

        sense_data1 = [x, np.array([sense_FOV, initial_R])]
        sense_data2 = [y, np.array([sense_FOV, initial_R])]

        # Update obstacle info based on sensed data
        obj1.update_obstacle_detection(x, low_real_obs, up_real_obs)
        obj2.update_obstacle_detection(y, low_real_obs, up_real_obs)
    else:
        raise ValueError('Unknown sensor type')

    print(f'Time Step: {t + 1}')

    # Compute avoid set
    obj1.compute_avoid_set(sense_data1, sense_shape, t + 1, x)
    obj2.compute_avoid_set(sense_data2, sense_shape, t + 1, y)

    # Update plot
    plt_obj.update_plot(x, obj1)
    plt_obj.update_plot(y, obj2)
    plt.draw()
    time.sleep(dt)

# Save output data
if save_output_data:
    np.savez(filename, valueFunCellArr=obj1.value_fun_cell_arr, lxCellArr=obj1.lx_cell_arr, QSizeCellArr=obj1.q_size_cell_arr,
             solnTimes=obj1.soln_times, fovCellArr=obj1.fov_cell_arr)
