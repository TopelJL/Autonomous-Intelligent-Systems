import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
import matplotlib.path as mpath
from mpl_toolkits.mplot3d import Axes3D

class Plotter:
    """
    A visualization class for plotting level sets, environment, and trajectories.
    Based on the RVL lab's backwards reachability implementation.
    """
    
    def __init__(self, low_env, up_env, low_obs, up_obs, obs_shape):
        """
        Initialize the plotter with environment and obstacle parameters.
        
        Args:
            low_env (np.ndarray): Lower (x,y) corner of environment
            up_env (np.ndarray): Upper (x,y) corner of environment
            low_obs (np.ndarray): Lower corner/center of obstacle
            up_obs (np.ndarray): Upper corner/radius of obstacle
            obs_shape (str): Either 'rectangle' or 'circle'
        """
        self.low_env = np.array(low_env)
        self.up_env = np.array(up_env)
        self.low_obs = np.array(low_obs)
        self.up_obs = np.array(up_obs)
        self.obs_shape = obs_shape
        
        # Initialize figure handles
        self.fig = None
        self.ax = None
        self.env_handle = None
        self.sense_handle = None
        self.car_handle = []
        self.vx_handle = []
        self.first_plot = True
        
    def update_plot(self, x, set_obj):
        """
        Update visualization with new state and value function data.
        
        Args:
            x (np.ndarray): Current state [x, y] or [x, y, theta]
            set_obj: Object containing grid and value function data
        """
        # Initialize or clear figure
        if self.first_plot:
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111)
            self.first_plot = False
        else:
            self._clear_handles()
        
        # Update all plot elements
        self.env_handle = self.plot_environment()
        self.sense_handle = self.plot_sensing(set_obj.g_fmm, set_obj.union_l_2d_fmm)
        self.car_handle = self.plot_car(x)
        self.plot_boundary_padding(set_obj.bound_low, set_obj.bound_up)
        
        # Plot value function
        extra_args = {'edge_color': 'r'}
        if len(x) == 3:
            extra_args['theta'] = x[2]
            value_func = set_obj.value_fun[:,:,:,-1]
        else:
            value_func = set_obj.value_fun[:,:,-1]
            
        self.vx_handle = self.plot_func_level_set(
            set_obj.grid, value_func, vis_set=True, extra_args=extra_args)
        
        plt.draw()
        
    def _clear_handles(self):
        """Clear all plot handles."""
        if self.env_handle:
            self.env_handle.remove()
        if self.sense_handle:
            self.sense_handle.remove()
        for handle in self.car_handle:
            if handle:
                handle.remove()
        for handle in self.vx_handle:
            if handle:
                handle.remove()
        self.car_handle = []
        self.vx_handle = []
        
    def plot_environment(self):
        """
        Plot the environment boundaries and obstacle.
        
        Returns:
            matplotlib.artist.Artist: Handle to the obstacle patch
        """
        if self.obs_shape == 'rectangle':
            width = self.up_obs[0] - self.low_obs[0]
            height = self.up_obs[1] - self.low_obs[1]
            obstacle = Rectangle(
                self.low_obs, width, height,
                linestyle='--',
                linewidth=2.0,
                facecolor='none',
                edgecolor='k'
            )
        else:  # circle
            obstacle = Circle(
                self.low_obs, self.up_obs[0],
                linestyle='--',
                linewidth=2.0,
                facecolor='none',
                edgecolor='k'
            )
            
        self.ax.add_patch(obstacle)
        
        # Set plot limits and labels
        self.ax.set_xlim(self.low_env[0], self.up_env[0])
        self.ax.set_ylim(self.low_env[1], self.up_env[1])
        self.ax.set_xlabel('$p_x$', fontsize=12)
        self.ax.set_ylabel('$p_y$', fontsize=12)
        self.ax.tick_params(length=0)
        self.ax.grid(True, linestyle=':')
        
        return obstacle
        
    def plot_car(self, x):
        """
        Plot the car state (position and heading if applicable).
        
        Args:
            x (np.ndarray): State vector [x, y] or [x, y, theta]
            
        Returns:
            list: Handles to the plotted elements
        """
        handles = []
        
        # Plot position
        pos_handle = self.ax.plot(
            x[0], x[1], 'ko',
            markersize=8,
            markeredgecolor='k',
            markerfacecolor='k',
            zorder=5
        )[0]
        handles.append(pos_handle)
        
        # Plot heading arrow if state includes orientation
        if len(x) == 3:
            R = np.array([
                [np.cos(x[2]), -np.sin(x[2])],
                [np.sin(x[2]), np.cos(x[2])]
            ])
            heading_pt = R @ np.array([0.5, 0]) + x[:2]
            
            arrow_handle = self.ax.plot(
                [x[0], heading_pt[0]],
                [x[1], heading_pt[1]],
                'k-',
                linewidth=2,
                zorder=4
            )[0]
            handles.append(arrow_handle)
            
        return handles
        
    def plot_sensing(self, grid, signed_distance_map):
        """
        Visualize the sensing region using a filled contour.
        
        Args:
            grid: Grid object containing coordinate vectors
            signed_distance_map: 2D array of signed distances
            
        Returns:
            matplotlib.artist.Artist: Handle to the sensing region patch
        """
        # Create contour at zero level
        contour = self.ax.contour(
            grid.xs[0], grid.xs[1],
            signed_distance_map,
            levels=[0],
            colors='w'
        )
        
        # Extract and fill contour paths
        paths = []
        for collection in contour.collections:
            for path in collection.get_paths():
                paths.append(path.vertices)
                
        if paths:
            poly = plt.Polygon(
                paths[0],
                facecolor=[0, 0.2, 1],
                alpha=0.3,
                edgecolor='w',
                zorder=2
            )
            self.ax.add_patch(poly)
            return poly
        return None
        
    def plot_boundary_padding(self, bound_low, bound_up):
        """
        Plot the computational domain boundary.
        
        Args:
            bound_low (np.ndarray): Lower corner of boundary
            bound_up (np.ndarray): Upper corner of boundary
        """
        width = bound_up[0] - bound_low[0]
        height = bound_up[1] - bound_low[1]
        rect = Rectangle(
            bound_low, width, height,
            linestyle=':',
            linewidth=1.0,
            facecolor='none',
            edgecolor='k',
            zorder=1
        )
        self.ax.add_patch(rect)
        
    def plot_func_level_set(self, grid, func, vis_set=True, extra_args=None):
        """
        Plot level sets of a function.
        
        Args:
            grid: Grid object with coordinate vectors
            func: Value function data
            vis_set (bool): If True, plot 2D level set, else 3D surface
            extra_args (dict): Additional plotting arguments
            
        Returns:
            list: Handles to the plotted elements
        """
        if extra_args is None:
            extra_args = {}
            
        # Project to 2D slice if needed
        if 'theta' in extra_args:
            grid_plot, data_plot = self.proj(
                grid, func,
                [0, 0, 1],
                extra_args['theta']
            )
        else:
            grid_plot = grid
            data_plot = func
            
        handles = []
        if vis_set:
            # 2D level set visualization
            contour = self.ax.contour(
                grid_plot.xs[0], grid_plot.xs[1],
                data_plot,
                levels=[0],
                colors=extra_args.get('edge_color', 'r'),
                linewidths=2,
                zorder=3
            )
            handles.extend(contour.collections)
        else:
            # 3D surface visualization
            if not isinstance(self.ax, Axes3D):
                self.ax = self.fig.add_subplot(111, projection='3d')
            
            surf = self.ax.plot_surface(
                grid_plot.xs[0], grid_plot.xs[1],
                data_plot,
                cmap='viridis',
                alpha=0.7
            )
            handles.append(surf)
            
        return handles
        
    def plot_waypts(self, waypts, sim_width, sim_height):
        """
        Plot waypoints and connecting path.
        
        Args:
            waypts (list): List of waypoint coordinates
            sim_width (float): Simulation width
            sim_height (float): Simulation height
        """
        waypts = np.array(waypts)
        self.ax.plot(
            waypts[:, 0], waypts[:, 1],
            'k-',
            linewidth=1.5,
            zorder=3
        )
        self.ax.scatter(
            waypts[:, 0], waypts[:, 1],
            c='k',
            s=50,
            zorder=4
        )
        self.ax.set_xlim(0, sim_width)
        self.ax.set_ylim(0, sim_height)
        
    def proj(self, grid, data, dims, value):
        """
        Project high-dimensional data to a lower-dimensional slice.
        
        Args:
            grid: Grid object
            data: Value function data
            dims (list): Dimensions to project along
            value (float): Value at which to take slice
            
        Returns:
            tuple: (projected_grid, projected_data)
        """
        # Find closest grid point to desired value
        if len(dims) == 1:
            dim = dims[0]
            idx = np.argmin(np.abs(grid.xs[dim] - value))
            if dim == 2:  # theta dimension
                return grid, data[:, :, idx]
        
        # Default: return original grid and data
        return grid, data