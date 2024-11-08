# ---------------------------------------------------------------------
#                        test_convergence.py
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
# Ryan Rahrooh   9/27/24     Iniital Matlab to python conversion
# ---------------------------------------------------------------------
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from backwards_reachability.visualization import Plotter

# Clear plots and variables
plt.clf()

# Load debugging data
path = 'safe_navigation/data/stuckVxLxE005.mat'
debug_data = loadmat(path)

# Extract variables from loaded data (V(x), l(x), lxOld(x), stuckRect, oneStep)
data0 = debug_data['data0']
lx = debug_data['lx']
lx_old = debug_data['lxOld']
stuck_rect = debug_data['stuckRect']
one_step = debug_data['oneStep']
scheme_data = debug_data['schemeData']

# Set solver parameters
update_epsilon = 0.005
min_with = 'minVWithL'

extra_args = {
    'stopConverge': 0,
    'ignoreBoundary': 1,
    'targets': lx
}

# Time setup
t0 = 0
dt = 0.05
t_max = 100  # tMax = 100 means compute infinite-horizon solution
times = np.arange(t0, t_max + dt, dt)

# Setup environment bounds
low_env = np.array([0, 0])
up_env = np.array([10, 7])
obs_shape = 'rectangle'
low_real_obs = np.array([4, 1])
up_real_obs = np.array([7, 4])

# Create plotter
plt = Plotter(low_env, up_env, low_real_obs, up_real_obs, obs_shape)

# Run HJIPDE solver with local updates
data_out, tau, extra_outs = HJIPDE_solve_local(
    data0, lx_old, lx, update_epsilon, times, scheme_data, min_with, extra_args, plt
)

# Run the normal solver (commented section from original)
"""
# Normal solver setup
update_epsilon = 0.01
extra = {
    'visualize': {
        'valueSet': 1,
        'initialValueSet': 1,
        'figNum': 1,
        'deleteLastPlot': True,
        'plotData': {
            'plotData': 1,
            'plotDims': [1, 1, 0],
            'projpt': np.pi/2
        },
        'plotColorVS0': [1, 0, 0.7],  # color of initial value set
        'plotColorVS': [1, 0, 0],     # color of final value set
        'plotColorTS': [0, 0, 1],     # color of new cost function
        'plotColorlxOld': [0, 0, 0]   # color of prior cost function
    }
}

# Uncomment to run normal solver
# data_out, tau, extra_outs = HJIPDE_solve(lx, times, scheme_data, min_with, extra)
"""

# Optional: Add legend
<<<<<<< HEAD
# plt.legend(['old V(x)', 'old l(x)', 'new l(x)', 'new V(x)'])
=======
# plt.legend(['old V(x)', 'old l(x)', 'new l(x)', 'new V(x)'])
>>>>>>> 3a683ff1f2e69270c34ce516750bfd6788ab5c55
