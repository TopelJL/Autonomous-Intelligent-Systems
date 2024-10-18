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
# Ryan Rahrooh   9/27/24     Iniital Matlab to python conversion
# ---------------------------------------------------------------------

import math

def getControlDubins(t, curr_state, goal_state):
    """
    Computes the control for a Dubins car to move towards a goal state.

    Args:
        t: Current time (not used in this version, but could be useful for future time-varying controls).
        curr_state: Current state of the car (x, y, theta).
        goal_state: Goal state of the car (x, y, theta).

    Returns:
        u: Control input [v, omega], where:
            v: Linear velocity (forward speed).
            omega: Angular velocity (rate of rotation).
    """

    # Maximum allowable velocities
    max_v = 1.0      # Maximum linear velocity
    max_omega = 1.0  # Maximum angular velocity

    # Unpack current and goal states
    curr_x, curr_y, curr_theta = curr_state
    goal_x, goal_y, goal_theta = goal_state

    # Calculate differences in position (x, y)
    delta_x = goal_x - curr_x
    delta_y = goal_y - curr_y

    # Compute the desired heading (theta) using atan2 to go towards the goal
    desired_theta = math.atan2(delta_y, delta_x)

    # Calculate heading error (difference between current heading and desired heading)
    error_theta = wrap_to_pi(desired_theta - curr_theta)

    # Compute the angular velocity to correct the heading
    omega = max_omega * error_theta  # Proportional control for steering (omega)
    omega = max(min(omega, max_omega), -max_omega)  # Limit omega to the allowed range

    # If the heading error is small (vehicle is generally facing the goal), move at full speed
    if abs(error_theta) < math.pi / 6:
        v = max_v
    else:
        # If the vehicle needs to turn sharply, reduce its speed to allow for better turning control
        v = 0.5 * max_v

    # Return the computed control input [v, omega]
    return [v, omega]

def wrap_to_pi(angle):
    """
    Wraps an angle to the range [-pi, pi], which is useful for calculating heading differences.

    Args:
        angle: The angle to wrap, in radians.

    Returns:
        Wrapped angle in the range [-pi, pi].
    """

    # Continuously adjust the angle to be within the [-pi, pi] range
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle
