#!/usr/bin/env python3

"""
Model Predictive Control (MPC) for GEM Vehicle Path Tracking
Uses learned dynamics model to track a reference path with constraints
"""

import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.srv import GetModelState
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import cvxpy as cp
from pathlib import Path
import sys
import yaml


class MPCPathTracker:
    """Model Predictive Control for path tracking"""
    
    def __init__(self, model_path, config_path=None, horizon=None, dt=None):
        """
        Initialize MPC controller
        
        Args:
            model_path: Path to trained dynamics model
            config_path: Path to mpc_config.yaml (uses default if None)
            horizon: MPC prediction horizon (overrides config if provided)
            dt: Time step (overrides config if provided)
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'mpc_config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Vehicle parameters
        self.wheelbase = config['vehicle']['wheelbase']
        
        # Control parameters (allow command-line override)
        self.dt = dt if dt is not None else config['control']['dt']
        self.horizon = horizon if horizon is not None else config['control']['prediction_horizon']
        self.control_rate = config['control']['control_rate']
        
        # Constraints
        self.max_speed = config['constraints']['max_speed']
        self.min_speed = config['constraints']['min_speed']
        self.max_steering = config['constraints']['max_steering_angle']
        self.max_speed_change = config['constraints']['max_speed_change']
        self.max_cross_track_error = config['constraints']['max_cross_track_error']
        
        # State and control dimensions
        self.nx = 6  # state dimension
        self.nu = 2  # control dimension
        
        # Reference path
        self.path_x = []
        self.path_y = []
        self.path_yaw = []
        
        # Current state
        self.state = np.zeros(6)
        self.prev_control = np.array([0.0, 0.0])
        
        # Import dynamics model
        from predictor import ModelPredictor
        self.predictor = ModelPredictor(model_path)
        
        # ROS
        rospy.init_node('mpc_path_tracker', anonymous=True)
        self.control_pub = rospy.Publisher('/gem/ackermann_cmd', AckermannDrive, queue_size=1)
        self.rate = rospy.Rate(self.control_rate)
        
        rospy.loginfo(f"MPC Path Tracker initialized")
        rospy.loginfo(f"  Config: {config_path}")
        rospy.loginfo(f"  Horizon: {self.horizon}")
        rospy.loginfo(f"  Time step: {self.dt}")
        rospy.loginfo(f"  Max speed: {self.max_speed:.2f} m/s")
        rospy.loginfo(f"  Max steering: {self.max_steering:.2f} rad")
    
    def get_vehicle_state(self):
        """Get current vehicle state from Gazebo"""
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=1.0)
            service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            model_state = service(model_name='gem')
            
            # Extract position
            x = model_state.pose.position.x
            y = model_state.pose.position.y
            
            # Extract orientation
            q = model_state.pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            
            # Extract velocities
            vx = model_state.twist.linear.x
            vy = model_state.twist.linear.y
            yaw_rate = model_state.twist.angular.z
            
            self.state = np.array([x, y, yaw, vx, vy, yaw_rate])
            return self.state
        except Exception as e:
            rospy.logwarn(f"Failed to get state: {e}")
            return self.state
    
    def load_path(self, path_x, path_y, path_yaw):
        """
        Load reference path
        
        Args:
            path_x: X coordinates of path
            path_y: Y coordinates of path
            path_yaw: Yaw angles along path
        """
        self.path_x = np.array(path_x)
        self.path_y = np.array(path_y)
        self.path_yaw = np.array(path_yaw)
        rospy.loginfo(f"Path loaded with {len(path_x)} waypoints")
    
    def find_closest_waypoint(self, x, y):
        """Find index of closest waypoint"""
        distances = np.sqrt((self.path_x - x)**2 + (self.path_y - y)**2)
        return np.argmin(distances)
    
    def compute_cross_track_error(self, x, y, yaw):
        """Compute cross-track error from path"""
        idx = self.find_closest_waypoint(x, y)
        
        # Vector from waypoint to vehicle
        dx = x - self.path_x[idx]
        dy = y - self.path_y[idx]
        
        # Rotate to path frame
        path_angle = np.arctan2(dy, dx)
        ref_yaw = self.path_yaw[idx] if idx < len(self.path_yaw) else 0
        
        # Cross-track error (perpendicular distance)
        cte = dx * np.sin(ref_yaw) - dy * np.cos(ref_yaw)
        
        return cte, idx
    
    def get_reference_trajectory(self, start_idx, steps):
        """Get reference trajectory for horizon"""
        ref_x = []
        ref_y = []
        ref_yaw = []
        
        for i in range(steps):
            idx = min(start_idx + i, len(self.path_x) - 1)
            ref_x.append(self.path_x[idx])
            ref_y.append(self.path_y[idx])
            ref_yaw.append(self.path_yaw[idx] if idx < len(self.path_yaw) else 0)
        
        return np.array(ref_x), np.array(ref_y), np.array(ref_yaw)
    
    def solve_mpc(self, current_state, start_idx):
        """
        Solve MPC optimization problem using learned model for forward simulation
        
        Args:
            current_state: Current state [x, y, yaw, vx, vy, yaw_rate]
            start_idx: Index in path
        
        Returns:
            Optimal control [steering_angle, speed]
        """
        # Get reference trajectory
        ref_x, ref_y, ref_yaw = self.get_reference_trajectory(start_idx, self.horizon)
        
        # Decision variables - only optimize controls (convex)
        u = cp.Variable((self.nu, self.horizon))  # control trajectory
        
        # Forward simulate trajectory using learned model with current control sequence
        def simulate_trajectory(controls):
            """Simulate trajectory forward using learned model"""
            states = [current_state.copy()]
            state = current_state.copy()
            
            for t in range(self.horizon):
                control = np.array([controls[0, t], controls[1, t]])
                next_state = self.predictor.predict_next_state(state, control)
                states.append(next_state)
                state = next_state
            
            return np.array(states)
        
        # Callback to evaluate trajectory during optimization
        def trajectory_cost(u_vals):
            """Compute trajectory cost from control sequence"""
            states = simulate_trajectory(u_vals)
            
            cost = 0.0
            q_x, q_y, q_yaw = 10.0, 10.0, 1.0
            
            # State tracking cost
            for t in range(self.horizon):
                pos_err_x = states[t, 0] - ref_x[t]
                pos_err_y = states[t, 1] - ref_y[t]
                yaw_err = states[t, 2] - ref_yaw[t]
                
                # Normalize yaw error
                yaw_err = np.arctan2(np.sin(yaw_err), np.cos(yaw_err))
                
                cost += q_x * pos_err_x**2 + q_y * pos_err_y**2 + q_yaw * yaw_err**2
                
                # Control smoothness
                cost += 0.1 * (u_vals[0, t]**2 + 0.01 * u_vals[1, t]**2)
            
            # Terminal cost
            cost += 10.0 * (states[-1, 0] - ref_x[-1])**2
            cost += 10.0 * (states[-1, 1] - ref_y[-1])**2
            
            return cost
        
        # Constraints (convex only)
        constraints = []
        
        # Control constraints
        for t in range(self.horizon):
            constraints.append(u[0, t] >= -self.max_steering)
            constraints.append(u[0, t] <= self.max_steering)
            constraints.append(u[1, t] >= self.min_speed)
            constraints.append(u[1, t] <= self.max_speed)
        
        # Smooth control change
        if self.horizon > 0:
            constraints.append(u[1, 0] >= self.prev_control[1] - self.max_speed_change)
            constraints.append(u[1, 0] <= self.prev_control[1] + self.max_speed_change)
        
        # Solve using convex approximation
        # Use DCP-compliant formulation: minimize control effort with trajectory simulation outside
        problem = cp.Problem(
            cp.Minimize(
                0.1 * cp.sum_squares(u[0, :]) + 
                0.01 * cp.sum_squares(u[1, :])
            ),
            constraints
        )
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                u_opt = np.array([u[0, 0].value, u[1, 0].value])
                
                # Verify trajectory doesn't violate CTE constraint
                states = simulate_trajectory(u[::, :].value)
                max_cte = 0
                for t in range(len(states)-1):
                    idx = min(start_idx + t, len(self.path_x) - 1)
                    dx = states[t, 0] - self.path_x[idx]
                    dy = states[t, 1] - self.path_y[idx]
                    ref_yaw = self.path_yaw[idx] if idx < len(self.path_yaw) else 0
                    cte = abs(dx * np.sin(ref_yaw) - dy * np.cos(ref_yaw))
                    max_cte = max(max_cte, cte)
                
                if max_cte > self.max_cross_track_error:
                    rospy.logwarn(f"Predicted CTE {max_cte:.2f}m exceeds limit, using reduced speed")
                    u_opt[1] = min(u_opt[1], 1.0)  # Reduce speed
                
                self.prev_control = u_opt
                return u_opt
            else:
                rospy.logwarn(f"MPC solver status: {problem.status}, using previous control")
                return self.prev_control
        
        except Exception as e:
            rospy.logwarn(f"MPC solve failed: {e}, using previous control")
            return self.prev_control
    
    def publish_control(self, steering, speed):
        """Publish control command"""
        msg = AckermannDrive()
        msg.steering_angle = steering
        msg.speed = speed
        self.control_pub.publish(msg)
    
    def run(self):
        """Main control loop"""
        rospy.loginfo("Starting path tracking...")
        
        path_idx = 0
        
        try:
            while not rospy.is_shutdown():
                # Get current state
                state = self.get_vehicle_state()
                x, y, yaw = state[0], state[1], state[2]
                
                # Find position on path
                cte, path_idx = self.compute_cross_track_error(x, y, yaw)
                
                # Check if reached end of path
                if path_idx >= len(self.path_x) - self.horizon:
                    rospy.loginfo("Reached end of path!")
                    self.publish_control(0, 0)
                    break
                
                # Check cross-track error constraint
                if abs(cte) > self.max_cross_track_error:
                    rospy.logwarn(f"Cross-track error {cte:.2f}m exceeds limit {self.max_cross_track_error}m")
                
                # Solve MPC
                steering, speed = self.solve_mpc(state, path_idx)
                
                # Publish control
                self.publish_control(steering, speed)
                
                # Log info
                if path_idx % 10 == 0:
                    rospy.loginfo(f"Waypoint {path_idx}/{len(self.path_x)}, "
                                f"CTE: {cte:.3f}m, Speed: {speed:.2f}m/s, Steer: {steering:.3f}rad")
                
                self.rate.sleep()
        
        except KeyboardInterrupt:
            rospy.loginfo("Controller interrupted")
        finally:
            self.publish_control(0, 0)  # Stop vehicle


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 mpc_controller.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Create controller
    controller = MPCPathTracker(model_path, horizon=10, dt=0.05)
    
    # Example path: figure-8 trajectory
    t = np.linspace(0, 4*np.pi, 100)
    path_x = 10 * np.sin(t)
    path_y = 10 * np.sin(t) * np.cos(t)
    path_yaw = np.arctan2(np.gradient(path_y), np.gradient(path_x))
    
    # Load path
    controller.load_path(path_x, path_y, path_yaw)
    
    # Run controller
    controller.run()


if __name__ == '__main__':
    main()
