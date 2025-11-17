#!/usr/bin/env python3

"""
CSV Path Tracker - Execute path from CSV file using MPC
Reads collected trajectory data and executes it as reference path
"""

import numpy as np
import pandas as pd
import rospy
from ackermann_msgs.msg import AckermannDrive
from gazebo_msgs.srv import GetModelState
from tf.transformations import euler_from_quaternion
import cvxpy as cp
from pathlib import Path
import sys
import yaml
import matplotlib.pyplot as plt
from datetime import datetime


class CSVPathTracker:
    """Execute path tracking from CSV data file"""
    
    def __init__(self, csv_path, model_path=None, config_path=None, horizon=None, dt=None):
        """
        Initialize path tracker from CSV
        
        Args:
            csv_path: Path to CSV file with trajectory data
            model_path: Path to trained dynamics model (optional)
            config_path: Path to mpc_config.yaml (uses default if None)
            horizon: MPC prediction horizon (overrides config if provided)
            dt: Time step (overrides config if provided)
        """
        self.csv_path = csv_path
        
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
        self.max_steering = config['constraints']['max_steering_angle']
        self.max_cross_track_error = config['constraints']['max_cross_track_error']
        
        # Load path from CSV
        self.load_path_from_csv(csv_path)
        
        # Load dynamics model if provided
        self.predictor = None
        if model_path:
            try:
                from predictor import ModelPredictor
                self.predictor = ModelPredictor(model_path)
                rospy.loginfo(f"Loaded dynamics model from {model_path}")
            except Exception as e:
                rospy.logwarn(f"Failed to load model: {e}")
        
        # State
        self.state = np.zeros(6)
        self.prev_control = np.array([0.0, 0.0])
        
        # Tracking for plots
        self.cte_history = []
        self.time_history = []
        self.start_time = None
        
        # ROS
        rospy.init_node('csv_path_tracker', anonymous=True)
        self.control_pub = rospy.Publisher('/gem/ackermann_cmd', AckermannDrive, queue_size=1)
        self.rate = rospy.Rate(self.control_rate)
        
        rospy.loginfo(f"CSV Path Tracker initialized")
        rospy.loginfo(f"  Config: {config_path}")
        rospy.loginfo(f"  Path length: {len(self.path_x)} waypoints")
        rospy.loginfo(f"  Distance: {self.path_length:.2f}m")
    
    def load_path_from_csv(self, csv_path):
        """Load path from CSV file - supports 5-column format: x, y, yaw, steering, speed"""
        try:
            df = pd.read_csv(csv_path, header=None)
            
            # 5-column format: x, y, yaw, steering_angle, speed
            if df.shape[1] == 5:
                self.path_x = df.iloc[:, 0].values
                self.path_y = df.iloc[:, 1].values
                self.path_yaw = df.iloc[:, 2].values
                self.path_steering = df.iloc[:, 3].values
                self.path_speed = df.iloc[:, 4].values
            # Alternative: assume x, y, yaw columns if named
            else:
                try:
                    self.path_x = df['x'].values
                    self.path_y = df['y'].values
                    self.path_yaw = df['yaw'].values
                    self.path_steering = np.zeros(len(self.path_x))
                    self.path_speed = np.ones(len(self.path_x)) * self.max_speed
                except KeyError:
                    raise ValueError("CSV must have columns: x, y, yaw or be 5-column format")
            
            # Compute path length
            diffs = np.diff(np.column_stack([self.path_x, self.path_y]), axis=0)
            distances = np.sqrt(np.sum(diffs**2, axis=1))
            self.path_length = np.sum(distances)
            
            rospy.loginfo(f"Loaded path from {csv_path}")
            rospy.loginfo(f"  Waypoints: {len(self.path_x)}")
            rospy.loginfo(f"  X range: [{self.path_x.min():.4f}, {self.path_x.max():.4f}]")
            rospy.loginfo(f"  Y range: [{self.path_y.min():.4f}, {self.path_y.max():.4f}]")
            rospy.loginfo(f"  Path length: {self.path_length:.4f}m")
            rospy.loginfo(f"  Speed range: [{self.path_speed.min():.4f}, {self.path_speed.max():.4f}] m/s")
        
        except Exception as e:
            rospy.logerr(f"Failed to load CSV: {e}")
            raise
    
    def get_vehicle_state(self):
        """Get current vehicle state"""
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=1.0)
            service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            model_state = service(model_name='gem')
            
            x = model_state.pose.position.x
            y = model_state.pose.position.y
            
            q = model_state.pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            
            vx = model_state.twist.linear.x
            vy = model_state.twist.linear.y
            yaw_rate = model_state.twist.angular.z
            
            self.state = np.array([x, y, yaw, vx, vy, yaw_rate])
            return self.state
        except Exception as e:
            rospy.logwarn(f"Failed to get state: {e}")
            return self.state
    
    def find_closest_waypoint(self, x, y):
        """Find closest waypoint index"""
        distances = np.sqrt((self.path_x - x)**2 + (self.path_y - y)**2)
        return np.argmin(distances)
    
    def compute_cross_track_error(self, x, y, yaw):
        """Compute cross-track error"""
        idx = self.find_closest_waypoint(x, y)
        
        dx = x - self.path_x[idx]
        dy = y - self.path_y[idx]
        
        ref_yaw = self.path_yaw[idx]
        
        cte = dx * np.sin(ref_yaw) - dy * np.cos(ref_yaw)
        
        return cte, idx
    
    def get_reference_trajectory(self, start_idx, steps):
        """Get reference trajectory with proper indexing"""
        ref_x = []
        ref_y = []
        ref_yaw = []
        ref_speed = []
        
        for i in range(steps):
            idx = min(start_idx + i, len(self.path_x) - 1)
            ref_x.append(self.path_x[idx])
            ref_y.append(self.path_y[idx])
            ref_yaw.append(self.path_yaw[idx])
            ref_speed.append(self.path_speed[idx])
        
        return np.array(ref_x), np.array(ref_y), np.array(ref_yaw), np.array(ref_speed)
    
    def solve_mpc_simple(self, current_state, start_idx):
        """
        MPC using learned dynamics model for trajectory prediction
        Optimizes steering and speed to track reference path
        """
        ref_x, ref_y, ref_yaw, ref_speed = self.get_reference_trajectory(start_idx, self.horizon)
        
        # Use learned model for MPC if available, otherwise use kinematic model
        if self.predictor is not None:
            return self._mpc_with_learned_model(current_state, ref_x, ref_y, ref_speed)
        else:
            return self._mpc_kinematic(current_state, ref_x, ref_y, ref_speed)
    
    def _mpc_with_learned_model(self, current_state, ref_x, ref_y, ref_speed):
        """
        MPC with learned dynamics model prediction
        Simulates trajectory and optimizes control to minimize tracking error
        """
        # Simulate multiple control sequences and pick the best one
        best_steering = 0.0
        best_speed = ref_speed[0] if len(ref_speed) > 0 else 2.0
        min_error = float('inf')
        
        # Grid search over steering and speed
        for trial_steering in np.linspace(-self.max_steering, self.max_steering, 7):
            for trial_speed_mult in np.linspace(0.5, 1.0, 3):
                trial_speed = np.clip(ref_speed[0] * trial_speed_mult, -1.0, self.max_speed)
                
                try:
                    # Simulate trajectory with this control
                    state = current_state.copy()
                    error = 0.0
                    
                    for h in range(min(self.horizon, len(ref_x))):
                        control = np.array([trial_steering, trial_speed])
                        
                        # Predict next state using learned model
                        state = self.predictor.predict_next_state(state, control)
                        
                        # Compute error to reference
                        pos_error = np.sqrt((state[0] - ref_x[h])**2 + (state[1] - ref_y[h])**2)
                        yaw_error = np.abs(self._normalize_angle(state[2] - ref_yaw[h]))
                        
                        error += pos_error + 0.5 * yaw_error
                    
                    # Control regularization (prefer less steering)
                    error += 0.1 * np.abs(trial_steering)
                    
                    if error < min_error:
                        min_error = error
                        best_steering = trial_steering
                        best_speed = trial_speed
                
                except Exception as e:
                    continue
        
        return np.array([best_steering, best_speed])
    
    def _mpc_kinematic(self, current_state, ref_x, ref_y, ref_speed):
        """
        Kinematic model-based MPC fallback
        Uses bicycle model without learned dynamics
        """
        current_x, current_y, current_yaw = current_state[0], current_state[1], current_state[2]
        
        # Look ahead to find target direction
        lookahead_idx = min(3, len(ref_x) - 1)
        target_x = ref_x[lookahead_idx]
        target_y = ref_y[lookahead_idx]
        
        # Compute desired heading
        dx = target_x - current_x
        dy = target_y - current_y
        desired_heading = np.arctan2(dy, dx)
        
        # Normalize angle difference
        heading_error = self._normalize_angle(desired_heading - current_yaw)
        
        # Proportional steering control (steering follows heading error)
        kp_steering = 1.0
        steering = np.clip(kp_steering * heading_error, -self.max_steering, self.max_steering)
        
        # Speed control (track reference speed)
        speed = np.clip(ref_speed[0] if len(ref_speed) > 0 else 2.0, -1.0, self.max_speed)
        
        return np.array([steering, speed])
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def publish_control(self, steering, speed):
        """Publish control command"""
        msg = AckermannDrive()
        msg.steering_angle = steering
        msg.speed = speed
        self.control_pub.publish(msg)
    
    def run(self):
        """Main control loop"""
        rospy.loginfo("=" * 60)
        rospy.loginfo("CSV PATH TRACKING STARTED")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Path waypoints: {len(self.path_x)}")
        rospy.loginfo(f"X range: [{self.path_x.min():.4f}, {self.path_x.max():.4f}] m")
        rospy.loginfo(f"Y range: [{self.path_y.min():.4f}, {self.path_y.max():.4f}] m")
        rospy.loginfo(f"Speed range: [{self.path_speed.min():.4f}, {self.path_speed.max():.4f}] m/s")
        rospy.loginfo(f"MPC horizon: {self.horizon} steps, dt: {self.dt}s")
        rospy.loginfo("=" * 60)
        
        path_idx = 0
        total_distance = 0
        step_count = 0
        max_cte = 0
        avg_cte = []
        self.start_time = rospy.Time.now().to_sec()
        
        try:
            while not rospy.is_shutdown():
                # Get current state
                state = self.get_vehicle_state()
                x, y, yaw = state[0], state[1], state[2]
                
                # Find position on path
                cte, path_idx = self.compute_cross_track_error(x, y, yaw)
                avg_cte.append(abs(cte))
                max_cte = max(max_cte, abs(cte))
                
                # Track CTE and time for plotting
                elapsed_time = rospy.Time.now().to_sec() - self.start_time
                self.cte_history.append(cte)
                self.time_history.append(elapsed_time)
                
                # Check if reached end
                if path_idx >= len(self.path_x) - self.horizon:
                    rospy.loginfo("=" * 60)
                    rospy.loginfo("REACHED END OF PATH!")
                    rospy.loginfo(f"Total waypoints traversed: {path_idx}")
                    rospy.loginfo(f"Steps executed: {step_count}")
                    rospy.loginfo(f"Max CTE: {max_cte:.4f}m (limit: {self.max_cross_track_error}m)")
                    rospy.loginfo(f"Avg CTE: {np.mean(avg_cte):.4f}m")
                    rospy.loginfo("=" * 60)
                    self.publish_control(0, 0)
                    break
                
                # Warn if cross-track error too large
                if abs(cte) > self.max_cross_track_error:
                    rospy.logwarn(f"CTE: {cte:.4f}m (limit: {self.max_cross_track_error}m)")
                
                # Solve MPC
                steering, speed = self.solve_mpc_simple(state, path_idx)
                self.prev_control = np.array([steering, speed])
                
                # Publish control
                self.publish_control(steering, speed)
                
                # Update distance
                if path_idx > 0:
                    total_distance = np.sum(np.sqrt(
                        np.diff(self.path_x[:path_idx+1])**2 + 
                        np.diff(self.path_y[:path_idx+1])**2
                    ))
                
                step_count += 1
                
                # Log progress every 20 steps
                if step_count % 20 == 0:
                    rospy.loginfo(
                        f"Step {step_count}: Waypoint {path_idx}/{len(self.path_x)}, "
                        f"CTE: {cte:.4f}m, Distance: {total_distance:.4f}m, "
                        f"Control: [δ={steering:.4f} rad, v={speed:.4f} m/s]"
                    )
                
                self.rate.sleep()
        
        except KeyboardInterrupt:
            rospy.loginfo("Path tracking interrupted by user")
        except Exception as e:
            rospy.logerr(f"Path tracking error: {e}")
        finally:
            self.publish_control(0, 0)
            
            # Save CTE plot
            if len(self.cte_history) > 0:
                self.save_cte_plot()
            
            rospy.loginfo("Path tracker stopped")
    
    def save_cte_plot(self):
        """Save cross-track error plot for demonstration video"""
        try:
            # Find workspace root
            script_path = Path(__file__).resolve()
            workspace_root = script_path.parent.parent.parent.parent.parent
            plots_dir = workspace_root / "data" / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(self.time_history, self.cte_history, 'b-', linewidth=1.5, label='Cross-Track Error')
            ax.axhline(y=self.max_cross_track_error, color='r', linestyle='--', 
                      linewidth=2, label=f'Max CTE Limit (±{self.max_cross_track_error}m)')
            ax.axhline(y=-self.max_cross_track_error, color='r', linestyle='--', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Cross-Track Error (m)', fontsize=12)
            ax.set_title('Path Tracking Performance: Cross-Track Error Over Time', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=10)
            
            # Add statistics box
            stats_text = f"Mean |CTE|: {np.mean(np.abs(self.cte_history)):.4f} m\n"
            stats_text += f"Max |CTE|: {np.max(np.abs(self.cte_history)):.4f} m\n"
            stats_text += f"Std Dev: {np.std(self.cte_history):.4f} m\n"
            stats_text += f"Duration: {self.time_history[-1]:.2f} s"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # Save with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = plots_dir / f"cte_over_time_{timestamp}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            rospy.loginfo(f"CTE plot saved to: {output_path}")
            
        except Exception as e:
            rospy.logwarn(f"Failed to save CTE plot: {e}")


def main():
    if len(sys.argv) < 2:
        print("\n" + "=" * 70)
        print("CSV PATH TRACKER - Execute MPC path tracking from CSV waypoints")
        print("=" * 70)
        print("\nUsage:")
        print("  python3 csv_path_tracker.py <csv_path> [model_path] [horizon] [dt]")
        print("\nArguments:")
        print("  csv_path    : Path to CSV file (5 cols: x, y, yaw, steering, speed)")
        print("  model_path  : Path to trained model (optional, uses kinematic if omitted)")
        print("  horizon     : MPC prediction horizon (default: 10)")
        print("  dt          : Time step in seconds (default: 0.05)")
        print("\nExample:")
        print("  python3 csv_path_tracker.py /path/to/wps.csv model.pth 10 0.05")
        print("\nConstraints:")
        print(f"  Max speed        : 20 km/h (5.56 m/s)")
        print(f"  Max steering     : ±0.5 rad (±28.6°)")
        print(f"  Max cross-track  : ±1.0 m")
        print("=" * 70 + "\n")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    dt = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
    
    # Verify CSV exists
    if not Path(csv_path).exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Create tracker
    try:
        tracker = CSVPathTracker(csv_path, model_path, horizon=horizon, dt=dt)
        tracker.run()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
