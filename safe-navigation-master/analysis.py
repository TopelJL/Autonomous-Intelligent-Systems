# ---------------------------------------------------------------------
#                             analysis.py
#                    University of Central Florida
#              Autonomous & Intelligent Systems Labratory
#
# Description: 
#
# Note(1): This research done was derived off of a paper by 
# Andrea Bajcsy, Somil Bansal, Eli Bronstein, Varun Tolani, and 
# Claire J. Tomlin.
#
# Note(2) Please see "An_Efficient_Reachability-Based_Framework_
# for_Provably_Safe_Autonomous_Navigation_in_Unknown_Environments"
# Inside of our documents folder for original work done.
#
# Author        Date        Description
# ------        ----        ------------
# Jaxon Topel   9/27/24     Initial Matlab to python conversion
# Ryan Rahroo   9/27/24     Iniital Matlab to python conversion
# ---------------------------------------------------------------------


import numpy as np
import scipy.io as sio

# Choose which method you wanna analyze: 'HJI' or 'localQ'
method = 'HJI'
warm = True

# Grab the data.
path = '/home/abajcsy/hybrid_ws/src/safe_navigation/data/'
if method == 'HJI':
    if warm:
        method_name = 'Warm Start HJI-VI'
        file_path = path + 'HJIwarm.mat'
        data = sio.loadmat(file_path)
        color = [0.1, 0.1, 1.0]
    else:
        method_name = 'HJI-VI'
        file_path = path + 'HJI.mat'
        data = sio.loadmat(file_path)
        color = [0, 0, 0]
elif method == 'localQ':
    method_name = 'Local Update Algorithm'
    file_path = path + 'localQwarm.mat'
    data = sio.loadmat(file_path)
    color = [1.0, 0.1, 0.1]
else:
    raise ValueError('Unsupported method type.')

# Load variables from the .mat file
valueFunCellArr = data['valueFunCellArr']
lxCellArr = data['lxCellArr']
QSizeCellArr = data['QSizeCellArr']
solnTimes = data['solnTimes']

# Compute avg num states updated
total_per_step = []
for q_size in QSizeCellArr:
    total_this_step = np.sum(q_size)
    total_per_step.append(total_this_step)
avg_num_states_updated = np.mean(total_per_step)

# Printing
print(f"-------- {method_name} --------")
avg_total_compute_time = np.mean((solnTimes % 1) * 24 * 3600)
print(f"avg total compute time (s): {avg_total_compute_time}")
print(f"avg num states updated: {avg_num_states_updated}")

# Load grid.mat for the conservative states computation
grid_data = sio.loadmat('grid.mat')

# Compute num conservative states
if method == 'HJI' and not warm:
    num_conserv_states = 0
else:
    other_value_funs = valueFunCellArr
    file_path = path + 'HJI.mat'
    gt_data = sio.loadmat(file_path)
    gt_value_funs = gt_data['valueFunCellArr']
    epsilon = 0.01
    num_conserv_states = how_conservative(other_value_funs, gt_value_funs, grid_data['grid'], epsilon)

print(f"avg num conservative states: {np.mean(num_conserv_states)}")


def how_conservative(other_value_funs, gt_value_funs, grid, epsilon):
    """Check how conservative the other value functions are wrt ground truth."""
    num_conserv_states = []
    for local_fun, gt_fun in zip(other_value_funs, gt_value_funs):
        # We want indicies to be empty (for local update to be more conservative)
        loc_states = (local_fun < -epsilon) & (gt_fun > epsilon)
        indices = np.where(loc_states)
        num_conserv_states.append(len(indices[0]))  # indices[0] contains the indices of true loc_states
    return num_conserv_states
