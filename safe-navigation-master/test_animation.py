# ---------------------------------------------------------------------
#                          test_animation.py
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
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
import imageio.v2 as imageio
from scipy.io import loadmat

class BackwardsReachNavigator:
    def __init__(self):
        # Environment setup
        self.low_env = np.array([0., 0.])
        self.up_env = np.array([10., 7.])
        self.pd_dims = 3
        
        # Grid setup
        self.N = np.array([31, 31, 21])
        self.grid_low = np.concatenate([self.low_env, [-np.pi]])
        self.grid_up = np.concatenate([self.up_env, [np.pi]])
        
        # Obstacle setup
        self.obs_shape = 'rectangle'
        self.low_real_obs = np.array([4., 1.])
        self.up_real_obs = np.array([7., 4.])
        
        # System parameters
        self.x_init = np.array([2.0, 2.5, np.pi/2])
        self.w_max = 1.0
        self.v_range = np.array([0.5, 1.0])
        self.dt = 0.05
        
        # Sensing parameters
        self.sense_shape = 'circle'
        self.sense_rad = 1.5
        
        # Visualization parameters
        self.cmap_hot = 'hot'
        
        # State and grid initialization
        self.initialize_system()
        
    def initialize_system(self):
        """Initialize the grid and dynamical system"""
        self.grid = self.create_grid()
        self.dyn_sys = self.create_dynamic_system()
        
    def create_grid(self) -> Dict:
        """Create a grid structure following RVL lab conventions"""
        grid = {
            'min': self.grid_low,
            'max': self.grid_up,
            'N': self.N,
            'dx': np.zeros(self.pd_dims)
        }
        
        # Calculate grid spacing
        for dim in range(self.pd_dims):
            grid['dx'][dim] = (grid['max'][dim] - grid['min'][dim]) / (grid['N'][dim] - 1)
            
        # Create vectors for each dimension
        grid['vs'] = []
        for dim in range(self.pd_dims):
            grid['vs'].append(
                np.linspace(grid['min'][dim], grid['max'][dim], grid['N'][dim])
            )
            
        return grid
    
    def create_dynamic_system(self) -> Dict:
        """Create the plane dynamical system"""
        return {
            'x': self.x_init.copy(),
            'w_max': self.w_max,
            'v_range': self.v_range,
            'update': self.update_state
        }
    
    def update_state(self, u: np.ndarray, dt: float, x: np.ndarray) -> np.ndarray:
        """Update the system state using plane dynamics
        
        Args:
            u: Control input [v, w]
            dt: Time step
            x: Current state [x, y, θ]
            
        Returns:
            Updated state
        """
        v = u[0]  # Linear velocity
        w = u[1]  # Angular velocity
        
        # Plane dynamics
        x_new = x + dt * np.array([
            v * np.cos(x[2]),
            v * np.sin(x[2]),
            w
        ])
        
        return x_new
    
    def get_control(self, value_function: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Compute optimal control based on the value function gradient
        
        Args:
            value_function: Current value function
            current_state: Current state of the system
            
        Returns:
            Optimal control input [v, w]
        """
        # Get gradient of value function at current state
        grad = self.compute_gradient(value_function, current_state)
        
        # Compute optimal control (implement your specific control law here)
        v = self.v_range[1]  # Default to maximum velocity
        w = -self.w_max * np.sign(grad[2])  # Basic gradient descent on heading
        
        return np.array([v, w])
    
    def compute_gradient(self, value_function: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Compute the gradient of the value function at the given state
        
        Args:
            value_function: Value function array
            state: State to compute gradient at
            
        Returns:
            Gradient vector
        """
        # Convert state to grid indices
        indices = self.state_to_indices(state)
        
        # Compute numerical gradient (central difference)
        grad = np.zeros(self.pd_dims)
        for dim in range(self.pd_dims):
            if indices[dim] > 0 and indices[dim] < self.N[dim] - 1:
                forward_idx = list(indices)
                backward_idx = list(indices)
                forward_idx[dim] += 1
                backward_idx[dim] -= 1
                grad[dim] = (value_function[tuple(forward_idx)] - 
                           value_function[tuple(backward_idx)]) / (2 * self.grid['dx'][dim])
                
        return grad
    
    def state_to_indices(self, state: np.ndarray) -> Tuple[int, ...]:
        """Convert continuous state to grid indices"""
        indices = []
        for dim in range(self.pd_dims):
            idx = int((state[dim] - self.grid['min'][dim]) / self.grid['dx'][dim])
            idx = np.clip(idx, 0, self.N[dim] - 1)
            indices.append(idx)
        return tuple(indices)
    
    def run_simulation(self, value_fun_path: str, save_gif: bool = True):
        """Run the main simulation loop
        
        Args:
            value_fun_path: Path to the value function data
            save_gif: Whether to save the visualization as a GIF
        """
        # Load value functions
        value_fun_data = loadmat(value_fun_path)
        value_fun_cell_arr = value_fun_data['valueFunCellArr']
        
        # Setup visualization
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(-15, 32)
        ax.set_zlim([-3, 3])
        
        frames = []
        
        # Main simulation loop
        for i in range(len(value_fun_cell_arr)):
            value_fun = value_fun_cell_arr[i]
            
            # Get control input and update state
            u = self.get_control(value_fun[:,:,:,-1], self.dyn_sys['x'])
            self.dyn_sys['x'] = self.dyn_sys['update'](u, self.dt, self.dyn_sys['x'])
            
            # Visualization
            self.visualize_step(ax, value_fun, self.dyn_sys['x'])
            
            if save_gif:
                # Capture frame
                plt.draw()
                frame = self.capture_frame()
                frames.append(frame)
            
            plt.pause(0.05)
            
        if save_gif:
            imageio.mimsave('simulation.gif', frames, duration=0.1)
            
    def visualize_step(self, ax, value_fun: np.ndarray, state: np.ndarray):
        """Visualize current simulation step"""
        ax.clear()
        
        # Plot value function level set
        self.plot_value_function(ax, value_fun[:,:,:,-1], state[2])
        
        # Plot obstacles
        self.plot_obstacles(ax)
        
        # Plot vehicle and sensing radius
        self.plot_vehicle(ax, state)
        self.plot_sensing_radius(ax, state)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Value')
        
    def capture_frame(self):
        """Capture current figure as an image for GIF creation"""
        fig = plt.gcf()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image

if __name__ == "__main__":
    # Create navigator instance
    navigator = BackwardsReachNavigator()
    
    # Run simulation
    value_fun_path = os.path.join('safe_navigation', 'data', 'groundTruthValueFuns.mat')
    navigator.run_simulation(value_fun_path)
