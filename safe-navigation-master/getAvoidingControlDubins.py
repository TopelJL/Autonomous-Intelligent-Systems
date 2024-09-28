# ---------------------------------------------------------------------
#                     getAvoidingControlDubins.py
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

import numpy as np
import getControlDubins as controller

def getAvoidingControlDubins(t, x, goal, avoid_obj):
    """
    Computes the control for a Dubins vehicle to reach a goal while avoiding obstacles.

    Args:
        t: Current time.
        x: Vehicle state vector (position, heading).
        goal: Goal position.
        avoid_obj: Obstacle avoidance object.

    Returns:
        u: Control input.
    """

    pos = x[:2]
    theta = x[2]

    # Compute control to steer toward the goal.
    u_goal = controller.getControlDubins(t, x, goal)

    # Calculate the distance to the obstacle's avoid set.
    avoid_dist = avoid_obj.getMinDistanceToObstacle(pos)

    # If the vehicle is too close to the obstacle, adjust the control.
    if avoid_dist < 0.5:
        # Add a repulsive force or control term to avoid the obstacle.
        u_avoid = avoid_obj.getAvoidanceControl(pos, theta)

        # Blend goal-seeking control with obstacle avoidance control.
        u = u_goal + u_avoid  # Modify as needed to balance the two.
    else:
        u = u_goal

    return u