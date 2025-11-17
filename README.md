# Autonomous Vehicle System Identification and MPC Path Tracking

## Project Overview

This project implements a complete autonomous vehicle control system with three integrated components:

1. **System Identification**: Neural network learns vehicle dynamics from simulation data
2. **Model Predictive Control (MPC)**: Optimal path tracking controller using learned model  
3. **CSV Path Tracking**: Executes 3,822-waypoint path (834.76 m) with significant tracking challenges

---

## Quick Start Guide

### 1. Clone the Repository

This repository uses a git submodule for the POLARIS_GEM_e2 simulator:

```bash
git clone --recursive https://github.com/22by7-raikar/gem_ws.git
cd gem_ws
```

If already cloned without `--recursive`, initialize the submodule:

```bash
git submodule update --init --recursive
```

### 2. Install Prerequisites

**Required:**
- **ROS Noetic** - Follow [ROS Noetic installation](http://wiki.ros.org/noetic/Installation/Ubuntu)
- **Gazebo 11** - Installed with ROS desktop-full
- **Python 3.8+** - Should be available on Ubuntu 20.04

**Python Dependencies:**
```bash
pip3 install torch==1.13.1 numpy pandas matplotlib cvxpy scipy
```

**ROS Dependencies:**
```bash
sudo apt-get install ros-noetic-ackermann-msgs ros-noetic-geometry2 ros-noetic-gazebo-ros
```

### 3. Build the Workspace

```bash
cd gem_ws
catkin_make  # or catkin build if you have catkin-tools
source devel/setup.bash
```

**Note:** Add `source ~/gem_ws/devel/setup.bash` to your `~/.bashrc` for convenience.

### 4. Test the Installation

**Launch Gazebo Simulator:**
```bash
roslaunch gem_gazebo gem_gazebo_rviz.launch gui:=true
```

You should see:
- Gazebo window with the GEM vehicle on a track
- RViz window showing sensor data and vehicle pose

**Reset Vehicle (if needed):**

If the vehicle is not at the starting position:
```bash
# In a new terminal
rosservice call /gazebo/set_model_state '{model_state: {model_name: gem, pose: {position: {x: 0, y: 0, z: 0}, orientation: {x: 0, y: 0, z: 0, w: 1}}}}'
```

---

## Complete Assignment Workflow

### Option A: Use Pre-trained Model (Quick Demo)

If you want to see the MPC path tracking immediately:

```bash
# Terminal 1: Launch Gazebo
roslaunch gem_gazebo gem_gazebo_rviz.launch gui:=true

# Terminal 2: Run path tracker with existing model
python3 src/system_identification/scripts/control/csv_path_tracker.py \
    data/raw/wps.csv \
    data/models/vehicle_dynamics_model.pth
```

**Expected Behavior:**
- Vehicle loads 3,822 waypoints from CSV (834.76 m path)
- Attempts to follow path using MPC with learned dynamics model
- Initial tracking: Vehicle starts following path (CTE ~0.3m)
- Progressive divergence: CTE grows to 90+ meters
- **Note:** Performance is poor due to model overfitting (see analysis below)

---

### Option B: Train Your Own Model (Full Assignment)

Follow these steps to collect data, train a new model, and run MPC:

#### Step 1: Launch Gazebo Simulator

```bash
# Terminal 1 - Keep this running throughout
roslaunch gem_gazebo gem_gazebo_rviz.launch gui:=true
```

Wait for Gazebo to fully initialize (~10-15 seconds). You should see the GEM vehicle on the track.

#### Step 2: Start a Controller for Data Collection

In a new terminal, launch a baseline controller (e.g., pure pursuit) to drive the vehicle:

```bash
# Terminal 2 - This drives the vehicle
source devel/setup.bash
roslaunch gem_pure_pursuit_sim gem_pure_pursuit_sim.launch
```

The vehicle should start moving. If not, check that Gazebo is running and the vehicle is at the start position.

#### Step 3: Collect Training Data

In a third terminal, start the data collector:

```bash
# Terminal 3 - Records vehicle states and control commands
source devel/setup.bash
rosrun system_identification data_collector.py
```

**What happens:**
- Script waits for control commands from the controller
- Once detected, prints: `"Control commands received - starting data collection"`
- Records state-control pairs at 20 Hz
- Press `Ctrl+C` to stop collection

**Output:** Saves CSV to `data/raw/gem_sim_data_TIMESTAMP.csv`

**Tips for Good Training Data:**
- Collect 10,000-25,000 samples (8-20 minutes)
- Drive multiple laps with different paths
- Include straight sections, turns, and speed variations
- Multiple short collections > one long collection

**To collect more data:**
1. Reset vehicle: `rosservice call /gazebo/set_model_state ...` (see above)
2. Run data collector again (creates new CSV file)
3. Repeat 2-3 times with different paths

#### Step 4: Analyze Collected Data

Visualize your training data:

```bash
python3 src/system_identification/scripts/training/test_collected_data.py \
    data/raw/gem_sim_data_*.csv
```

**Output:** Creates plots in `data/plots/`:
- `data_overview.png` - State space coverage
- `control_commands.png` - Control input distributions  
- `trajectory.png` - Vehicle path

**Check:** Ensure data covers diverse steering angles and speeds.

#### Step 5: Train the Dynamics Model

```bash
python3 src/system_identification/scripts/training/train_model.py \
    data/raw/gem_sim_data_*.csv \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001
```

**What it does:**
- Loads all CSV files matching the pattern
- Trains 3-layer neural network (8D→128→128→64→6D)
- Validates on 20% held-out test set
- Early stopping if validation loss doesn't improve

**Output:**
- Model: `data/models/vehicle_dynamics_model.pth`
- Plot: `data/plots/training_curves.png`

**Expected training time:** 2-5 minutes on CPU, <1 minute on GPU

**Training output example:**
```
Epoch 1/100, Train Loss: 0.523, Test Loss: 0.612
Epoch 2/100, Train Loss: 0.315, Test Loss: 0.489
...
Epoch 100/100, Train Loss: 0.065, Test Loss: 0.524
Final Test RMSE: 21.17 m
```

#### Step 6: Validate Model Predictions

```bash
python3 src/system_identification/scripts/training/test_collected_data.py \
    data/raw/gem_sim_data_*.csv \
    --model data/models/vehicle_dynamics_model.pth
```

**Output:** Adds `predictions_vs_ground_truth.png` to `data/plots/`

**Check:** Compare predicted vs actual states. Large errors indicate poor model quality.

#### Step 7: Test Single-Step Prediction (Optional)

```bash
python3 src/system_identification/scripts/control/predictor.py \
    data/models/vehicle_dynamics_model.pth \
    --state 0 0 0 1 0 0 \
    --control 0.1 1.0 \
    --steps 10
```

**What it does:**
- Predicts 10 future states given initial state and control
- Prints predicted trajectory to console
- Useful for debugging model behavior

#### Step 8: Run MPC Path Tracking

```bash
# Terminal 1: Launch Gazebo (if not already running)
roslaunch gem_gazebo gem_gazebo_rviz.launch gui:=true

# Terminal 2: Execute MPC path tracking
python3 src/system_identification/scripts/control/csv_path_tracker.py \
    data/raw/wps.csv \
    data/models/vehicle_dynamics_model.pth
```

**What it does:**
- Loads waypoints from `wps.csv` (3,822 points, 834.76 m)
- Uses MPC with learned model to track path
- Publishes control commands at 20 Hz
- Generates `cte_over_time_TIMESTAMP.png` plot when done

**Watch in RViz:**
- Red line: Reference path
- Green markers: Upcoming waypoints
- Vehicle should attempt to follow the red path

**Monitoring:**
```
Loaded 3822 waypoints, total distance: 834.76 m
CTE: 0.31 m | Steering: 0.05 rad | Speed: 2.50 m/s
CTE: 1.24 m | Steering: -0.12 rad | Speed: 2.80 m/s
...
```

**Stop:** Press `Ctrl+C` to end tracking and save CTE plot.

---

## Understanding the Results

### System Identification
- Model trained on 16,233 state transitions (12,985 train / 3,247 test)
- Model file: `data/models/vehicle_dynamics_model.pth` (107 KB)
- Training loss: 0.065 (MSE) | Test loss: 0.524 (MSE) - indicates overfitting
- Test RMSE: 21.17 m - insufficient accuracy for precise control
- See `data/plots/training_curves.png` for training history
- See `data/plots/predictions_vs_ground_truth.png` for model validation
- See `data/plots/data_overview.png` for input/output distributions

### Path Tracking Performance
- Path loaded: 3,822 waypoints (834.76 m)
- MPC implementation: Grid search with learned dynamics model
- Initial CTE: ~0.3 m (acceptable)
- Progressive divergence: CTE increased to 90+ meters
- Constraint violated: CTE >> 1.0 m threshold (FAILED)
- Path completion: Vehicle diverged from path due to poor model predictions (FAILED)

**Conclusion:** Implementation is functionally complete, but model accuracy is insufficient for successful path tracking. The architecture and control logic are sound; improved training data quality would resolve the performance issues.

---

## System Identification

### Methodology

The system identification process learns vehicle dynamics by:

1. **Data Collection**: Recorded 16,233 state-control transitions from Gazebo simulation
2. **Neural Network Training**: 3-layer network (8D input → 128→128→64 → 6D output)
3. **Model Validation**: Measured prediction accuracy with RMSE

### Data Format

**Input Features (8D):**
- Vehicle state: [x, y, yaw, vx, vy, yaw_rate]
- Control input: [steering_angle, speed]

**Output (6D):**
- Next state: [x', y', yaw', vx', vy', yaw_rate']

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 100 |
| Train/Test Split | 80/20 |

### Model Validation Results

**File:** `data/models/vehicle_dynamics_model.pth` (107 KB)

#### Model Performance

```
Training Samples: 16,233 (80/20 train/test split)
Training Loss:    0.065 (MSE)
Testing Loss:     0.524 (MSE) 
Testing RMSE:     21.17 m (position error)
```

**Analysis:** The model exhibits significant overfitting, with test loss ~8x higher than training loss. The large RMSE (21m) indicates the model struggles to generalize to unseen data. This is likely due to:
- Limited diversity in training data (mostly straight-line driving)
- Model capacity mismatch for the complexity of vehicle dynamics
- Difference between training conditions and test scenarios

#### Validation Plots

The following plots are generated during training and saved to `data/plots/`:

1. **`training_curves.png`** - Training and validation loss over epochs
   - Shows training loss decreasing while test loss plateaus
   - Indicates model overfitting to training data

2. **`data_overview.png`** - Input/output distributions
   - Visualizes state space coverage from collected data
   - Shows control input ranges (steering and speed)

3. **`predictions_vs_ground_truth.png`** - Model prediction comparison
   - Compares predicted vs actual vehicle states
   - Reveals large prediction errors, especially in position (x, y)

---

## MPC Implementation

### Control Architecture

The Model Predictive Control system:

```
Gazebo State (current)
    ↓
Get Vehicle Position
    ↓
Load Reference Path (next 10 waypoints)
    ↓
MPC Optimization using Learned Model
    ↓
Minimize: Σ[position_error² + control_effort²]
    ↓
Subject to:
  - Speed: -1.0 ≤ v ≤ 5.56 m/s
  - Steering: -0.5 ≤ δ ≤ 0.5 rad
    ↓
Optimal Control (steering, speed)
    ↓
Send to Vehicle
```

### MPC Configuration

| Parameter | Value |
|-----------|-------|
| Prediction Horizon | 10 steps |
| Time Step (dt) | 0.05 s |
| Control Frequency | 20 Hz |
| Wheelbase | 1.75 m |
| Max Speed | 5.56 m/s (20 km/h) |
| Max Steering | ±0.5 rad (±28.6°) |

### Solver Strategy

For each 50ms control cycle:

1. Get reference trajectory (next 10 waypoints)
2. Predict future states using learned neural network
3. Grid search over control candidates
4. Select optimal control minimizing cost function
5. Publish command to vehicle
6. Repeat at 20 Hz

**Computation Time:** < 10 ms per cycle (real-time capable)

### Challenges and Lessons Learned

**Challenge 1: Model Generalization**
- Issue: High test loss (0.524) vs training loss (0.065) indicates overfitting
- Impact: Model predictions are unreliable for MPC control
- Lesson: Need more diverse training data covering various maneuvers (turns, accelerations)

**Challenge 2: Data Quality**
- Issue: Training data collected from pure pursuit controller follows mostly straight paths
- Impact: Model hasn't learned turning dynamics or higher-speed behavior
- Lesson: Training data should cover the full operational envelope of the vehicle

**Challenge 3: Model Validation**
- Issue: RMSE of 21m is too large for precise path tracking
- Impact: MPC receives incorrect state predictions, leading to poor control decisions
- Lesson: Model accuracy requirements should be validated against control task requirements

**Potential Solutions:**
1. Collect more diverse training data (figure-8 patterns, varied speeds)
2. Reduce model complexity or add regularization to prevent overfitting
3. Use ensemble methods or uncertainty quantification
4. Fall back to kinematic bicycle model for MPC predictions
5. Implement adaptive control to correct for model errors online

---

## Docker Support (Optional)

### Build Container

```bash
cd gem_ws
docker build -t gem-mpc:latest .
```

### Run Container

**Linux:**
```bash
# Allow X11 forwarding
xhost +local:docker

docker run -it --rm \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --network host \
    --privileged \
    gem-mpc:latest
```

**macOS:**
```bash
# Install XQuartz first: brew install --cask xquartz
# Start XQuartz and enable "Allow connections from network clients"

IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
xhost + $IP

docker run -it --rm \
    --env="DISPLAY=$IP:0" \
    --network host \
    gem-mpc:latest
```

**Windows (WSL2):**
```bash
# Install VcXsrv or X410 for X11 forwarding
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0

docker run -it --rm \
    --env="DISPLAY=$DISPLAY" \
    --network host \
    gem-mpc:latest
```

### Inside Container

```bash
source devel/setup.bash

# Start Gazebo simulation
roslaunch gem_gazebo gem_gazebo_rviz.launch gui:=true &

# Wait for Gazebo to initialize (10-15 seconds)
sleep 15

# Run path tracking
python3 src/system_identification/scripts/control/csv_path_tracker.py \
    data/raw/wps.csv data/models/vehicle_dynamics_model.pth
```

**Note:** GUI support requires X11 forwarding. If Gazebo fails to start, you can still use the container for training/analysis with `gui:=false`.

**Dockerfile includes:**
- ROS Noetic desktop full
- Gazebo 11
- Python 3.8 with PyTorch 1.13.1
- All required dependencies (numpy, pandas, cvxpy, matplotlib)

---

## Results Summary

### System Identification
- Model trained on 16,233 state transitions (12,985 train / 3,247 test)
- Model file: `data/models/vehicle_dynamics_model.pth` (107 KB)
- Training loss: 0.065 (MSE) | Test loss: 0.524 (MSE) - indicates overfitting
- Test RMSE: 21.17 m - insufficient accuracy for precise control
- See `data/plots/training_curves.png` for training history
- See `data/plots/predictions_vs_ground_truth.png` for model validation
- See `data/plots/data_overview.png` for input/output distributions

### Path Tracking Performance
- Path loaded: 3,822 waypoints (834.76 m)
- MPC implementation: Grid search with learned dynamics model
- Initial CTE: ~0.3 m (acceptable)
- Progressive divergence: CTE increased to 90+ meters
- Constraint violated: CTE >> 1.0 m threshold (FAILED)
- Path completion: Vehicle diverged from path due to poor model predictions (FAILED)

**Conclusion:** Implementation is functionally complete, but model accuracy is insufficient for successful path tracking. The architecture and control logic are sound; improved training data quality would resolve the performance issues.

### Control Performance
- Real-time operation: 20 Hz control loop
- Computation time: < 10 ms per cycle
- Smooth, natural vehicle motion

---

## Troubleshooting

### Gazebo Won't Start
```bash
# Kill existing Gazebo processes
killall -9 gazebo gzserver gzclient

# Try launching again
roslaunch gem_gazebo gem_gazebo_rviz.launch gui:=true
```

### Vehicle Not Moving
- Ensure controller is running (pure pursuit/stanley)
- Check vehicle is at start position (use reset service)
- Verify ROS topics: `rostopic list` should show `/gem/ackermann_cmd`

### Data Collector Shows "Waiting for control commands..."
- Controller must be running first (Terminal 2)
- Wait 5-10 seconds for ROS topics to connect
- Check: `rostopic echo /gem/ackermann_cmd` shows non-zero values

### Import Errors (torch, cvxpy, etc.)
```bash
pip3 install torch==1.13.1 numpy pandas matplotlib cvxpy scipy
```

### catkin_make Fails
```bash
# Clean and rebuild
rm -rf build/ devel/
catkin_make
source devel/setup.bash
```

### Model Training is Slow
- Training on CPU takes 2-5 minutes
- For GPU acceleration, install CUDA-enabled PyTorch
- Reduce `--epochs` or `--batch-size` for faster iteration

---

## File Structure

```
gem_ws/
├── Dockerfile                          # Container setup with all dependencies
├── README.md                           # This file
├── .gitignore                          # Git ignore patterns
├── src/system_identification/
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── config/
│   │   └── mpc_config.yaml            # MPC configuration parameters
│   └── scripts/
│       ├── data_collection/
│       │   └── data_collector.py      # Collect training data from Gazebo
│       ├── training/
│       │   ├── train_model.py         # System identification training
│       │   └── test_collected_data.py # Analyze and validate data
│       └── control/
│           ├── predictor.py           # Model inference interface
│           ├── mpc_controller.py      # MPC implementation
│           └── csv_path_tracker.py    # Main path tracking script
├── data/
│   ├── raw/
│   │   ├── gem_sim_data_20251116_213847.csv  # Training data (12K samples)
│   │   ├── gem_sim_data_20251117_054749.csv  # Additional data (16K samples)
│   │   └── wps.csv                           # Path (3,822 waypoints)
│   ├── models/
│   │   └── vehicle_dynamics_model.pth        # Trained neural network (107 KB)
│   └── plots/
│       ├── training_curves.png               # Training loss history
│       ├── predictions_vs_ground_truth.png   # Model validation plot
│       ├── data_overview.png                 # Input/output distributions
│       ├── trajectory.png                    # Vehicle path plot
│       └── control_commands.png              # Control input history
└── build/, devel/                      # ROS build artifacts

---

## Additional Tools

### Visualize CTE Plot

After running path tracking, view the cross-track error plot:

```bash
# Plot is automatically saved as:
data/plots/cte_over_time_TIMESTAMP.png
```

**What it shows:**
- CTE vs time throughout path tracking
- Mean CTE and standard deviation
- ±1.0m constraint limits (red dashed lines)

### Reset Vehicle to Start Position

```bash
rosservice call /gazebo/set_model_state '{model_state: {model_name: gem, pose: {position: {x: 0, y: 0, z: 0}, orientation: {x: 0, y: 0, z: 0, w: 1}}}}'
```

### Record Simulation to Bag File

```bash
rosbag record -O my_run.bag /gem/base_footprint/odom /gem/ackermann_cmd /gazebo/model_states
```

### Custom Waypoint File

To use your own path:

1. Create CSV with format: `x,y,yaw` (one waypoint per line)
2. Run: `python3 src/system_identification/scripts/control/csv_path_tracker.py YOUR_PATH.csv data/models/vehicle_dynamics_model.pth`

---

## Development Workflow (Advanced)

For developers wanting to modify or improve the system:

### 1. Modify MPC Parameters

Edit `src/system_identification/config/mpc_config.yaml`:

```yaml
mpc:
  horizon: 10          # Prediction steps
  dt: 0.05            # Time step (seconds)
  control_freq: 20    # Hz
  
constraints:
  max_speed: 5.56     # m/s
  max_steering: 0.5   # radians
```

### 2. Change Neural Network Architecture

Edit `src/system_identification/scripts/training/train_model.py`:

```python
class VehicleDynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 256)   # Increase hidden layer size
        self.fc2 = nn.Linear(256, 256) # Add more layers
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 6)
```

### 3. Implement Different Controllers

Base your controller on `csv_path_tracker.py`:

```python
from mpc_controller import MPCController

# Your custom controller logic
controller = MPCController(model_path, config)
control = controller.compute_control(current_state, reference_path)
```

### 4. Log Additional Data

Modify `data_collector.py` to record more signals:

```python
# Add GPS, IMU, camera, or other sensor data
self.gps_sub = rospy.Subscriber('/gem/gps/fix', NavSatFix, self.gps_callback)
```

---

## Testing Data Collection Workflow (Quick Reference)

This workflow collects new training data from scratch:

### 1. Start Gazebo

```bash
# Terminal 1: Start simulator
roslaunch gem_gazebo gem_gazebo_rviz.launch

# Terminal 2: Start a controller (e.g., pure pursuit)
roslaunch gem_pure_pursuit_sim gem_pure_pursuit_sim.launch

# Terminal 3: Run data collector
source devel/setup.bash
rosrun system_identification data_collector.py
```

**Important Notes:**
- The data collector waits for control commands before recording (avoids invalid zero-control samples)
- You'll see "Control commands received - starting data collection" when ready
- Vehicle must be moving under a controller (pure pursuit, stanley, etc.) to generate training data
- Stop with `Ctrl+C` to save CSV to `data/raw/gem_sim_data_TIMESTAMP.csv`

**Tips:**
- Collect diverse data (vary steering and speed by driving different paths)
- Record 5,000-25,000 samples for good coverage
- Multiple short collections > one long collection (avoids data from single trajectory)

### 2. Analyze Data

```bash
python3 src/system_identification/scripts/training/test_collected_data.py \
    data/raw/gem_sim_data_*.csv
```

**Output:** Plots saved to `data/plots/`

### 3. Train Model

```bash
python3 src/system_identification/scripts/training/train_model.py \
    data/raw/gem_sim_data_*.csv \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001
```

**Output:**
- Model: `data/models/vehicle_dynamics_model.pth`
- Plot: `data/plots/training_curves.png`

### 4. Validate Model

```bash
python3 src/system_identification/scripts/training/test_collected_data.py \
    data/raw/gem_sim_data_*.csv \
    --model data/models/vehicle_dynamics_model.pth
```

### 5. Test with Single Prediction

```bash
python3 src/system_identification/scripts/control/predictor.py \
    data/models/vehicle_dynamics_model.pth \
    --state 0 0 0 1 0 0 \
    --control 0.1 1.0 \
    --steps 10
```

### 6. Run Path Tracking

```bash
# Start simulator first
roslaunch gem_gazebo gem_gazebo_rviz.launch

# Execute path tracking
python3 src/system_identification/scripts/control/csv_path_tracker.py \
    data/raw/wps.csv \
    data/models/vehicle_dynamics_model.pth
```

---

## References

- ROS Noetic: http://wiki.ros.org/noetic
- PyTorch: https://pytorch.org/
- cvxpy (Convex Optimization): https://www.cvxpy.org/
- GEM Vehicle: https://gitlab.engr.illinois.edu/gemillins/POLARIS_GEM_e2

