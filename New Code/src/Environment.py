# ---------------------------------------------------------------------
#                           Environment.py
#                    University of Central Florida
#              Autonomous & Intelligent Systems Labratory
#
# Description: 
#
#
# Author        Date        Description
# ------        ----        ------------
# Jaxon Topel   10/3/24     Initial Architecture Creation
# ---------------------------------------------------------------------

import numpy as np
import gym
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt


# Simulation environment for SLAM-based robot navigation.
class Environment(gym.Env):
    def __init__(self, config):
       
        # Args:
            # world_size (tuple): Size of the world (width, height)
            # robot_radius (float): Radius of the robot
            # sensor_range (float): Range of the robot's sensor (e.g., LiDAR)
            # num_landmarks (int): Number of landmarks in the environment
            # max_speed (float): Maximum speed of the robot
            # reward_goal (float): Reward for reaching the goal
            # reward_collision (float): Penalty for collision
            # reward_step (float): Reward for each successful step
            # goal_position (tuple): Optional starting position of the goal
            # obstacle_map (list): Optional pre-defined obstacle map
           
        # World parameters
        self.world_size = config["world_size"]
        self.robot_radius = config["robot_radius"]
        self.sensor_range = config["sensor_range"]

        # Robot and sensor parameters
        self.action_space = Discrete(4)  # Up, Down, Left, Right
        self.max_speed = config["max_speed"]

        # Reward parameters
        self.reward_goal = config["reward_goal"]
        self.reward_collision = config["reward_collision"]
        self.reward_step = config["reward_step"]

        # Goal and obstacle definitions
        self.goal_position = config.get("goal_position", None)
        self.obstacle_map = config.get("obstacle_map", None)

        # Reset the environment
        self.reset()

    def reset(self):
        """
        Reset the environment to a new initial state.

        Returns:
            observation (numpy.ndarray): Initial observation of the environment.
        """

        # Randomly place the robot within bounds
        self.robot_position = (
            self.robot_radius + np.random.rand(2) * (self.world_size[0] - 2 * self.robot_radius),
        )

        # Reset robot orientation (optional)
        self.robot_orientation = 0  # Can be expanded to represent heading

        # Generate a random map with obstacles (optional, comment out if using obstacle_map)
        # self.map = self._generate_map(...)

        # Observation: robot position and sensor data (replace with your implementation)
        observation = self._get_observation()

        return observation

    def step(self, action):
        """
        Take an action and update the environment.

        Args:
            action (int): Action from the action space (Up, Down, Left, Right)

        Returns:
            observation (numpy.ndarray): Observation after taking the action.
            reward (float): Reward for the action.
            done (bool): Whether the episode is finished.
            info (dict): Additional information about the step.
        """

        # Validate action (optional)
        assert self.action_space.contains(action), "Invalid action!"

        # Update robot position based on action and speed
        dx, dy = self._action_to_movement(action)
        new_position = self.robot_position + np.array([dx, dy]) * self.max_speed

        # Clip position within world bounds
        self.robot_position = np.clip(new_position, self.robot_radius, self.world_size - self.robot_radius)

        # Check for collisions with obstacles
        reward = self._get_collision_reward()
        done = reward == self.reward_collision

        # Check for reaching the goal (optional)
        if self.goal_position is not None and self._is_at_goal():
            reward = self.reward_goal
            done = True

        # Otherwise, reward for each successful step
        if not done:
            reward += self.reward_step

        # Observation: update with new robot position and sensor data
        observation = self._get_observation()

        info = {"position": self.robot_position}  # Add relevant info

        return observation, reward, done, info
    
    def visualize_environment(env):
        """
        Visualizes the current state of the environment.

        Args:
            env (SLAMEnv): The environment instance.
        """

        # Create a figure and axes
        fig, ax = plt.subplots()

        # Set plot limits based on world size
        ax.set_xlim(0, env.world_size[0])
        ax.set_ylim(0, env.world_size[1])

        # Plot the environment
        # ... (add code to plot obstacles, landmarks, etc.)

        # Plot the robot
        ax.plot(env.robot_position[0], env.robot_position[1], 'bo', markersize=10)

        # Plot the goal (if defined)
        if env.goal_position is not None:
            ax.plot(env.goal_position[0], env.goal_position[1], 'go', markersize=10)
            
        # Plot the map
        for obstacle in env.map:
            x, y = zip(*obstacle)
            ax.fill(x, y, 'gray', alpha=0.5)

        # Show the plot
        plt.show()