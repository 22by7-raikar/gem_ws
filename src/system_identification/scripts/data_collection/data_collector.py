#!/usr/bin/env python3

"""
Data Collector for GEM Vehicle System Identification
Records vehicle state and control commands from the simulator
Subscribes to: /gem/ackermann_cmd, /gazebo/get_model_state
Outputs: CSV file with state and control data
"""

import rospy
import numpy as np
import csv
import os
from pathlib import Path
from collections import deque
from datetime import datetime
from ackermann_msgs.msg import AckermannDrive
from gazebo_msgs.srv import GetModelState
from tf.transformations import euler_from_quaternion


class GEMDataCollector:
    def __init__(self, max_buffer=100000, collection_rate=50):
        """
        Initialize data collector for GEM vehicle
        
        Args:
            max_buffer: Maximum number of samples to store
            collection_rate: Frequency of data collection (Hz)
        """
        rospy.init_node('gem_data_collector', anonymous=True)
        
        # Configuration
        self.collection_rate = collection_rate
        self.rate = rospy.Rate(collection_rate)
        
        # State variables [x, y, yaw, vx, vy, yaw_rate]
        self.state = {
            'x': 0.0,
            'y': 0.0,
            'yaw': 0.0,
            'vx': 0.0,
            'vy': 0.0,
            'yaw_rate': 0.0,
        }
        
        # Previous state for velocity estimation
        self.prev_state = self.state.copy()
        self.prev_time = rospy.Time.now().to_sec()
        
        # Control inputs [steering_angle, speed]
        self.control = {
            'steering_angle': 0.0,
            'speed': 0.0,
        }
        self.control_received = False  # Flag to ensure we have valid control data
        
        # Data buffer
        self.data_buffer = deque(maxlen=max_buffer)
        self.wheelbase = 1.75  # meters (GEM vehicle spec)
        
        # Subscribers
        self.control_sub = rospy.Subscriber('/gem/ackermann_cmd', AckermannDrive, 
                                            self.control_callback, queue_size=1)
        
        rospy.loginfo("GEM Data Collector initialized")
        rospy.loginfo(f"Collection rate: {collection_rate} Hz")
        rospy.loginfo("Waiting for data... Press Ctrl+C to stop and save.")
    
    def get_gem_state(self):
        """
        Get current GEM vehicle state from Gazebo
        Returns: (x, y, yaw, vx, vy, yaw_rate) or None if service fails
        """
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=1.0)
            service_response = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            model_state = service_response(model_name='gem')
            
            # Extract position
            x = model_state.pose.position.x
            y = model_state.pose.position.y
            
            # Extract orientation (convert quaternion to yaw)
            orientation_q = model_state.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
            
            # Extract velocities from Twist
            vx = model_state.twist.linear.x
            vy = model_state.twist.linear.y
            yaw_rate = model_state.twist.angular.z
            
            return x, y, yaw, vx, vy, yaw_rate
            
        except rospy.ServiceException as e:
            rospy.logwarn(f"Service call failed: {e}")
            return None
    
    def control_callback(self, msg):
        """Record control commands from Ackermann message"""
        self.control['steering_angle'] = msg.steering_angle
        self.control['speed'] = msg.speed
        if not self.control_received:
            self.control_received = True
            rospy.loginfo("Control commands received - starting data collection")
    
    def record_data(self):
        """Get current state and record data point"""
        # Wait until we receive at least one control command
        if not self.control_received:
            return False
        
        state_tuple = self.get_gem_state()
        
        if state_tuple is None:
            return False
        
        x, y, yaw, vx, vy, yaw_rate = state_tuple
        
        # Update state dictionary
        self.state['x'] = x
        self.state['y'] = y
        self.state['yaw'] = yaw
        self.state['vx'] = vx
        self.state['vy'] = vy
        self.state['yaw_rate'] = yaw_rate
        
        # Create data point
        data_point = {
            'timestamp': rospy.Time.now().to_sec(),
            'x': self.state['x'],
            'y': self.state['y'],
            'yaw': self.state['yaw'],
            'vx': self.state['vx'],
            'vy': self.state['vy'],
            'yaw_rate': self.state['yaw_rate'],
            'steering_angle': self.control['steering_angle'],
            'speed': self.control['speed'],
        }
        
        self.data_buffer.append(data_point)
        return True
    
    def save_data(self, output_dir=None):
        """Save collected data to CSV file"""
        if len(self.data_buffer) == 0:
            rospy.logwarn("No data to save!")
            return None
        
        # Find workspace root and set output to data/raw/
        if output_dir is None:
            script_path = Path(__file__).resolve()
            workspace_root = script_path.parent.parent.parent.parent.parent
            output_dir = workspace_root / "data" / "raw"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f'gem_sim_data_{timestamp}.csv'
        
        # Write CSV
        keys = self.data_buffer[0].keys()
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.data_buffer)
            
            rospy.loginfo(f"Data saved to: {filename}")
            rospy.loginfo(f"  Total samples: {len(self.data_buffer)}")
            return filename
        except Exception as e:
            rospy.logerr(f"Failed to save data: {e}")
            return None
    
    def run(self):
        """Main collection loop"""
        sample_count = 0
        
        try:
            while not rospy.is_shutdown():
                if self.record_data():
                    sample_count += 1
                    if sample_count % 500 == 0:
                        rospy.loginfo(f"Collected {sample_count} samples...")
                
                self.rate.sleep()
        
        except KeyboardInterrupt:
            rospy.loginfo("\n" + "="*50)
            rospy.loginfo("Stopping data collection...")
            rospy.loginfo("="*50)
        
        finally:
            # Save data on exit
            rospy.loginfo(f"Total samples collected: {len(self.data_buffer)}")
            saved_file = self.save_data()
            
            if saved_file:
                rospy.loginfo("\nNext steps:")
                rospy.loginfo(f"1. Train the model:")
                rospy.loginfo(f"   python3 $(rospack find system_identification)/scripts/train_model.py {saved_file}")
                rospy.loginfo(f"2. Test predictions:")
                rospy.loginfo(f"   python3 $(rospack find system_identification)/scripts/test_collected_data.py {saved_file}")


def main():
    collector = GEMDataCollector(max_buffer=100000, collection_rate=50)
    collector.run()


if __name__ == '__main__':
    main()
