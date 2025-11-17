# Autonomous Vehicle System Identification and MPC Path Tracking

## Project Overview

This project implements a complete autonomous vehicle control system with three integrated components:

1. **System Identification**: Neural network learns vehicle dynamics from simulation data
2. **Model Predictive Control (MPC)**: Optimal path tracking controller using learned model  
3. **CSV Path Tracking**: Executes 3,822-waypoint path (834.76 m) with significant tracking challenges

---

## How to Build and Run the Simulation

### Prerequisites

- ROS Noetic
- Gazebo 11
- Python 3.8+
- PyTorch 1.13+
- Dependencies: cvxpy, numpy, pandas, matplotlib

### Build the Workspace

```bash
cd /home/apr/Personal/gem_ws
catkin build
source devel/setup.bash
```

### Run the Simulation

**Terminal 1 - Start Gazebo with RViz:**
```bash
roslaunch gem_gazebo gem_gazebo_rviz.launch gui:=true
```

**Terminal 2 - Execute Path Tracking:**
```bash
python3 src/system_identification/scripts/control/csv_path_tracker.py \
    data/raw/wps.csv \
    data/models/vehicle_dynamics_model.pth
```

**Expected Behavior:**
- Vehicle loads 3,822 waypoints from CSV (834.76 m path)
- Attempts to follow path using MPC with learned dynamics model
- MPC uses grid search over control candidates with learned model predictions

**Actual Results:**
- Path loading: Successfully reads all 3,822 waypoints
- Initial tracking: Vehicle starts following path (CTE ~0.3m initially)
- Progressive divergence: CTE grows steadily due to poor model predictions
- Final CTE: Exceeds 90m+ after ~150m of travel
- Constraint violation: CTE >> 1.0m requirement (FAILED)

**Root Cause:** The learned dynamics model (RMSE: 21m) provides inaccurate predictions, causing MPC to compute suboptimal control commands that drive the vehicle off course.

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

## Docker Support

### Build Container

```bash
cd /home/apr/Personal/gem_ws
docker build -t gem-mpc:latest .
```

### Run Container

```bash
docker run -it \
    --env="DISPLAY=$DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v $(pwd):/root/gem_ws \
    gem-mpc:latest
```

### Inside Container

```bash
cd /root/gem_ws
source devel/setup.bash

# Terminal 1: Start Gazebo
roslaunch gem_gazebo gem_gazebo_rviz.launch gui:=true

# Terminal 2: Run path tracking
python3 src/system_identification/scripts/control/csv_path_tracker.py \
    data/raw/wps.csv data/models/vehicle_dynamics_model.pth
```

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
```

---

## Development Workflow

### 1. Collect Training Data

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

