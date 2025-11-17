cmake_minimum_required(VERSION 3.0.2)
project(system_identification)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  ackermann_msgs
  gazebo_msgs
  tf
)

catkin_package()

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/data_collection/data_collector.py
  scripts/training/train_model.py
  scripts/training/test_collected_data.py
  scripts/control/predictor.py
  scripts/control/mpc_controller.py
  scripts/control/csv_path_tracker.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
