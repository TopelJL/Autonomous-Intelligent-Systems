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
# Ryan Rahrooh   9/27/24     Iniital Matlab to python conversion
# ---------------------------------------------------------------------

import numpy as np
import getControlDubins as controller

def getAvoidingControlDubins(t, x, goal, avoid_obj, avoid_threshold=0.5, alpha=0.8):
    """
    Computes the control for a Dubins vehicle to reach a goal while avoiding obstacles.

    Args:
        t: Current time.
        x: Vehicle state vector (position, heading).
        goal: Goal position.
        avoid_obj: Obstacle avoidance object (AvoidSet instance).
        avoid_threshold: Distance threshold to start obstacle avoidance.
        alpha: Weight for blending goal-seeking and obstacle avoidance controls.

    Returns:
        u: Control input.
    """

    pos = x[:2]  # Extract position (x, y)
    theta = x[2]  # Extract heading (theta)

    # Compute control to steer toward the goal.
    u_goal = controller.getControlDubins(t, x, goal)

    # Calculate the distance to the closest obstacle's avoid set.
    avoid_dist = avoid_obj.getMinDistanceToObstacle(pos)

    # Initialize control as goal-seeking control
    u = u_goal

    # If the vehicle is too close to the obstacle, adjust the control.
    if avoid_dist < avoid_threshold:
        # Get the obstacle avoidance control.
        u_avoid = avoid_obj.getAvoidanceControl(pos, theta)

        # Blend goal-seeking control with obstacle avoidance control.
        # alpha controls the importance of goal-seeking, and (1 - alpha) the importance of obstacle avoidance.
        u = alpha * u_goal + (1 - alpha) * u_avoid
    else:
        # No blending needed if obstacle is far enough; just use goal-seeking control.
        u = u_goal

    return u
