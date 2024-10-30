# ---------------------------------------------------------------------
#                            AvoidSet.py
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
from scipy.interpolate import RegularGridInterpolator
import skfmm  # Fast Marching Method library (requires installation)
from some_dynamics_module import Plane, KinVehicle2D  # Replace with actual dynamics models or imports

class AvoidSet:
    """
    AvoidSet computes and stores an avoid set, translated from MATLAB to Python.
    Includes Fast Marching Method (FMM) and Hamilton-Jacobi-Isaacs PDE solver placeholder.
    """
    
    def __init__(self, gridLow, gridUp, lowRealObs, upRealObs, obsShape, xinit, N, dt, 
                 updateEpsilon, warmStart, updateMethod, noiseMean, noiseStd, vehicleRadius):
        # Initialize variables with provided parameters
        self.gridLow = gridLow  
        self.gridUp = gridUp
        self.N = N  # Discretization points
        self.dt = dt  # Timestep for simulation
        self.updateEpsilon = updateEpsilon  # Convergence threshold for updates
        self.computeTimes = []  # Will store computation times for the avoid set calculation
        self.warmStart = warmStart  # Whether to use a warm start or not
        self.updateMethod = updateMethod  # Method to update the avoid set (HJI or localQ)
        self.noiseMean = noiseMean  # Gaussian noise mean for sensor data
        self.noiseStd = noiseStd  # Gaussian noise standard deviation for sensor data
        self.vehicleRadius = vehicleRadius  # Radius of the vehicle
        self.firstCompute = True  # Boolean flag for first-time computations
        
        # Determine if we have a periodic dimension (like in a Dubins car where heading wraps around)
        self.pdDims = 3 if len(N) == 3 else None  # MATLAB's pdDims is now handled by this condition
        
        # Create the grid based on lower and upper bounds, using a custom method create_grid (MATLAB: createGrid)
        self.grid = self.create_grid(self.gridLow, self.gridUp, self.N, self.pdDims)

        # Store obstacle coordinates (low and upper bounds of the real obstacle)
        self.lowRealObs = lowRealObs
        self.upRealObs = upRealObs
        
        # Create the 'ground-truth' cost function for the environment obstacle
        if obsShape == 'rectangle':
            self.lReal = self.shape_rectangle(self.grid, lowRealObs, upRealObs)
        elif obsShape == 'circle':
            center = lowRealObs
            radius = upRealObs[0]
            self.lReal = self.shape_cylinder(self.grid, 3, center, radius)
        else:
            raise ValueError(f"Unknown obstacle shape: {obsShape}")  # Error handling for unsupported shapes

        # Add a boundary padding to the computation domain, similar to MATLAB's boundary logic
        self.boundLow, self.boundUp = self.add_boundary_padding(self.gridLow, self.gridUp, self.N)
        lBoundary = -self.shape_rectangle(self.grid, self.boundLow, self.boundUp)
        self.lReal = self.shape_union(self.lReal, lBoundary)  # Merge with boundary obstacle

        # Current cost function representation (initially None in Python)
        self.lCurr = None
        
        # Initialize vehicle's dynamic system (Plane for Dubins car, KinVehicle2D for point mass)
        if len(N) == 3:
            wMax = 1.0
            vrange = [0.5, 1.0]
            self.dynSys = Plane(xinit, wMax, vrange)
        else:
            vMax = 1.0
            drift = 1.0
            self.dynSys = KinVehicle2D(xinit, vMax, drift)
        
        # Setup other necessary properties for value function computation
        self.timeDisc = np.arange(0, 100 + self.dt, self.dt)  # Discretized time vector
        self.uMode = 'max'  # Control is maximizing the value function
        
        # Scheme data and extra arguments (used for solving the HJ PDE)
        self.schemeData = {
            'grid': self.grid,
            'dynSys': self.dynSys,
            'accuracy': 'high',
            'uMode': self.uMode
        }
        self.HJIextraArgs = {
            'ignoreBoundary': 0,
            'stopConverge': int(self.updateMethod == 'HJI'),  # Stop if converged (HJI method only)
            'convergeThreshold': self.updateEpsilon if self.updateMethod == 'HJI' else None
        }

        # Variables to store the value function and other data
        self.valueFunCellArr = []  # Stores sequence of value functions
        self.lxCellArr = []  # Stores sequence of l(x) (cost function)
        self.QSizeCellArr = []  # Stores queue size for localQ method
        self.fovCellArr = []  # Stores results of the FMM (fast marching method)
        self.solnTimes = []  # Total time to compute solution at each step

    def create_grid(self, gridLow, gridUp, N, pdDims=None):
        """
        Create the computation grid.
        """
        return np.mgrid[
            gridLow[0]:gridUp[0]:complex(0, N[0]),
            gridLow[1]:gridUp[1]:complex(0, N[1]),
            gridLow[2]:gridUp[2]:complex(0, N[2])
        ]

    def shape_rectangle(self, grid, low, up):
        """
        Create a rectangle shape in the grid.
        """
        rect = np.logical_and(grid >= low, grid <= up)
        return rect

    def shape_cylinder(self, grid, dim, center, radius):
        """
        Create a cylindrical obstacle.
        """
        distances = np.sqrt((grid[0] - center[0])**2 + (grid[1] - center[1])**2)
        return distances <= radius

    def shape_union(self, shape1, shape2):
        """
        Union of two shapes (element-wise max).
        """
        return np.maximum(shape1, shape2)

    def add_boundary_padding(self, gridLow, gridUp, N):
        """
        Add padding to the grid boundaries.
        """
        offsetX = (gridUp[0] - gridLow[0]) / N[0]
        offsetY = (gridUp[1] - gridLow[1]) / N[1]
        if len(N) == 3:
            boundLow = [gridLow[0] + offsetX, gridLow[1] + offsetY, gridLow[2]]
            boundUp = [gridUp[0] - offsetX, gridUp[1] - offsetY, gridUp[2]]
        else:
            boundLow = [gridLow[0] + offsetX, gridLow[1] + offsetY]
            boundUp = [gridUp[0] - offsetX, gridUp[1] - offsetY]
        return boundLow, boundUp

    def add_gaussian_noise(self, data, vehicle="first"):
        """
        Add Gaussian noise to vehicle data.
        """
        noise = np.zeros_like(data)
        noise[:2] = self.noiseMean + self.noiseStd * np.random.randn(2)  # Apply Gaussian noise to the first two indices
        noisy_data = data + noise
        print(f'(x, y, heading) position of {vehicle} vehicle with noise: {noisy_data}')
        return noisy_data

    def getMinDistanceToObstacle(self, vehiclePosition):
        """
        Calculate the minimum distance between the vehicle and the obstacle.
        """
        minDist = np.inf  # Initialize minimum distance to a large value
        vehiclePosition = np.array(vehiclePosition).reshape(-1, 1)  # Ensure it's a column vector

        for i in range(self.lReal.shape[1]):
            avoidPoint = self.lReal[:2, i]
            dist = np.linalg.norm(vehiclePosition[:2] - avoidPoint)
            if dist < minDist:
                minDist = dist
        return minDist

    def getAvoidanceControl(self, vehiclePos, heading):
        """
        Compute control to steer away from an obstacle.
        """
        obsCenter = (self.lowRealObs + self.upRealObs) / 2
        avoidanceDir = vehiclePos - obsCenter
        avoidanceDir /= np.linalg.norm(avoidanceDir)
        angleAway = np.arctan2(avoidanceDir[1], avoidanceDir[0])
        u_avoid = angleAway - heading
        return np.clip(u_avoid, -np.pi/4, np.pi/4)

    def updateObstacleDetection(self, vehiclePos, lowRealObs, upRealObs):
        """
        Update the obstacle detection.
        """
        distanceToObs = self.getMinDistanceToObstacle(vehiclePos)
        if distanceToObs < 1.0:
            self.lowRealObs = lowRealObs + 0.1
            self.upRealObs = upRealObs - 0.1
        else:
            self.lowRealObs = lowRealObs
            self.upRealObs = upRealObs

    def compute_fmm_map(self, gFMM, occupancy_map):
        """
        Compute the Fast Marching Method (FMM) signed distance map.
        """
        return skfmm.distance(occupancy_map)

    def computeAvoidSet(self, senseData, senseShape, currTime, vehiclePositions):
        """
        Computes the avoid set based on sensor input and vehicle positions.
        """
        if senseShape == 'rectangle':
            lowSenseXY = senseData[0][:2]
            upSenseXY = senseData[1]
            lowObs = np.append(lowSenseXY, self.gridLow[2])
            upObs = np.append(upSenseXY, self.gridUp[2])
            sensingShape = -self.shape_rectangle(self.grid, lowObs, upObs)
        elif senseShape == 'circle':
            center = senseData[0][:2]
            radius = senseData[1][0]
            sensingShape = -self.shape_cylinder(self.grid, 3, center, radius)
        else:
            raise ValueError(f"Unsupported sense shape: {senseShape}")

        # Compute the union of the sensed region with the known obstacle
        unionL = self.shape_union(sensingShape, self.lReal)
        
        # Project the slice of unionL and create an occupancy map
        occupancy_map = np.sign(unionL)
        
        # Compute FMM to generate the signed distance function for the sensed region
        fmm_signed_dist = self.compute_fmm_map(self.grid, occupancy_map)
        self.unionL_2D_FMM = fmm_signed_dist  # Store the FMM results
        
        # Update the cost function lCurr
        if self.lCurr is None:
            self.lCurr = unionL
        else:
            self.lCurr = np.minimum(unionL, self.lCurr)

        # Now solve the HJI PDE if needed
        if self.updateMethod == 'HJI':
            data0 = self.lCurr if self.firstCompute else self.valueFunCellArr[-1]
            dataOut, tau, extraOuts = self.solve_HJIPDE(data0, self.lCurr, self.schemeData, self.timeDisc, 'minVWithL', self.HJIextraArgs)
            self.valueFunCellArr.append(dataOut)
            self.QSizeCellArr.append(extraOuts['QSizes'])
        
        return self.unionL_2D_FMM

    def solve_HJIPDE(self, data0, lCurr, schemeData, timeDisc, minWith, extraArgs):
        """
        Solves the HJI PDE (placeholder).
        """
        dataOut = data0
        tau = timeDisc
        extraOuts = {'QSizes': []}
        raise NotImplementedError("HJIPDE solver not implemented in Python.")
        return dataOut, tau, extraOuts

    def evaluate_value(self, grid, valueFun, x):
        """
        Evaluates the value function at a specific state x using interpolation.
        """
        interpolator = RegularGridInterpolator(grid, valueFun, bounds_error=False, fill_value=None)
        return interpolator(x)

    def computeGradients(self, grid, valueFun):
        """
        Computes the gradients of the value function on the grid.
        """
        gradients = np.gradient(valueFun, axis=tuple(range(len(grid))))
        return gradients
    
    def checkAndGetSafetyControl(self, x):
        """
        Checks if the vehicle is at the safety boundary and returns the optimal control.
        
        Args:
        - x: The current state of the vehicle.
        
        Returns:
        - uOpt: Optimal control to maintain safety.
        - onBoundary: Boolean indicating if the vehicle is on the safety boundary.
        """
        value = self.evaluate_value(self.grid, self.valueFunCellArr[-1], x)
        tol = 0.1  # Tolerance for boundary proximity
        
        # If the vehicle is close to the boundary
        if value < tol:
            deriv = self.computeGradients(self.grid, self.valueFunCellArr[-1])  # Compute gradient
            current_deriv = self.evaluate_value(self.grid, deriv, x)  # Get derivative at x
            uOpt = self.dynSys.optCtrl(x, current_deriv, self.uMode)  # Get optimal control from dynamics model
            return uOpt, True  # Vehicle is on or near the boundary
        
        return np.zeros(len(x)), False  # Return zero control if not on boundary

    def evaluate_value(self, grid, valueFun, x):
        """
        Evaluates the value function at a specific state x using interpolation.
        
        Args:
        - grid: The grid of points where the value function is defined.
        - valueFun: Array representing the value function on the grid.
        - x: The state at which to evaluate the value function.
        
        Returns:
        - Interpolated value of the value function at x.
        """
        interpolator = RegularGridInterpolator(grid, valueFun, bounds_error=False, fill_value=None)
        return interpolator(x)

   

