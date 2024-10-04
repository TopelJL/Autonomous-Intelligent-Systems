# ---------------------------------------------------------------------
#                        exactSolnHalfPlane.py
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
import matplotlib.pyplot as plt

N = 51
Nt = 51
L = 1
num_infty = 1000000

X = np.linspace(-L, L, N)
Y = np.linspace(-L, L, N)
TH = np.linspace(0, 2 * np.pi, Nt)

x, y, th = np.meshgrid(X, Y, TH)
xr = x  # x - y
yr = y  # x + y

rho = 0.3
ue = np.zeros((N, N, Nt))
turn_th1 = th
turn_th2 = (th + np.pi) % (2 * np.pi)
change_th1 = np.minimum(turn_th1, np.abs(turn_th1 - 2 * np.pi))
change_th2 = np.minimum(turn_th2, np.abs(turn_th2 - 2 * np.pi))
change_th = np.minimum(change_th1, change_th2)
move_horiz = xr - rho * np.sin(change_th)
ue = move_horiz + rho * change_th
ue[xr <= 0] = 0

ii = np.arange(1, N - 1)

plt.figure(1)
plt.clf()

final_value = ue[ii, ii, :]
plt.isosurface(x[ii, ii, :], y[ii, ii, :], th[ii, ii, :] * 180 / np.pi, final_value, 0.5)
plt.hold(True)

for kk in range(1, 11):
    plt.isosurface(x[ii, ii, :], y[ii, ii, :], th[ii, ii, :] * 180 / np.pi, final_value, 0.2 * kk)

# plt.isosurface(x[ii, ii, :], y[ii, ii, :], th[ii, ii, :] * 180 / np.pi, final_value, 0.4)
# plt.isosurface(x[ii, ii, :], y[ii, ii, :], th[ii, ii, :] * 180 / np.pi, final_value, num_infty / 2)

plt.xlabel('x')
plt.ylabel('y')
plt.axis([-L, L, -L, L, 0, 360])
plt.title(f'N = {N}, N_theta = {Nt}')
plt.axis('equal')

plt.show()