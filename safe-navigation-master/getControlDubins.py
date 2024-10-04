# ---------------------------------------------------------------------
#                          getControlDubins.py
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
# -----------   -------     -----------------------------------
# Jaxon Topel   9/27/24     Initial Matlab to python conversion
# Ryan Rahroo   9/27/24     Iniital Matlab to python conversion
# ---------------------------------------------------------------------

import math

def getControlDubins(t, curr_state, goal_state):
    """
    Computes the control for a Dubins car.

    Args:
        t: Current time.
        curr_state: Current state of the car (x, y, theta).
        goal_state: Goal state of the car (x, y, theta).

    Returns:
        u: Control input (velocity, angular velocity).
    """

    max_v = 1.0
    max_omega = 1.0

    curr_x, curr_y, curr_theta = curr_state
    goal_x, goal_y, goal_theta = goal_state

    delta_x = goal_x - curr_x
    delta_y = goal_y - curr_y
    desired_theta = math.atan2(delta_y, delta_x)

    error_theta = wrap_to_pi(desired_theta - curr_theta)
    omega = max_omega * error_theta
    omega = max(min(omega, max_omega), -max_omega)

    if abs(error_theta) < math.pi / 6:
        v = max_v
    else:
        v = 0.5 * max_v

    return [v, omega]

def wrap_to_pi(angle):
    """
    Wraps an angle to the range [-pi, pi].

    Args:
        angle: The angle to wrap.

    Returns:
        The wrapped angle.
    """

    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle