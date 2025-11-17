#!/bin/bash

# Quick start script for system identification workflow

echo "=================================================="
echo "GEM Vehicle System Identification - Quick Start"
echo "=================================================="
echo ""

WORKSPACE="/home/apr/Personal/gem_ws"
DATA_DIR="$WORKSPACE/data"

# Setup
echo "1. Setting up environment..."
cd "$WORKSPACE"
source devel/setup.bash
echo "✓ Environment ready"
echo ""

# Check simulator
echo "2. Checking simulator status..."
if rostopic list | grep -q "gem"; then
    echo "✓ Simulator is running"
else
    echo "✗ Simulator not detected. Start with:"
    echo "  roslaunch gem_gazebo gem_gazebo_rviz.launch"
    exit 1
fi
echo ""

# Data collection
echo "3. Starting data collection..."
echo "   [Press Ctrl+C when done collecting data]"
python3 "$WORKSPACE/devel/lib/system_identification/data_collector.py"

# Get the latest data file
LATEST_DATA=$(ls -t "$DATA_DIR"/gem_sim_data_*.csv 2>/dev/null | head -1)

if [ -z "$LATEST_DATA" ]; then
    echo "✗ No data file found"
    exit 1
fi

echo ""
echo "=================================================="
echo "Data collected: $LATEST_DATA"
echo "=================================================="
echo ""

# Analyze
echo "4. Analyzing collected data..."
python3 "$WORKSPACE/devel/lib/system_identification/test_collected_data.py" "$LATEST_DATA"
echo ""

# Train
echo "5. Training model..."
echo "   [This may take a few minutes...]"
python3 "$WORKSPACE/devel/lib/system_identification/train_model.py" "$LATEST_DATA" \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001

MODEL_FILE="$DATA_DIR/vehicle_dynamics_model.pth"

if [ ! -f "$MODEL_FILE" ]; then
    echo "✗ Model training failed"
    exit 1
fi

echo ""
echo "=================================================="
echo "Model trained: $MODEL_FILE"
echo "=================================================="
echo ""

# Evaluate
echo "6. Evaluating model predictions..."
python3 "$WORKSPACE/devel/lib/system_identification/test_collected_data.py" "$LATEST_DATA" \
    --model "$MODEL_FILE"

echo ""
echo "=================================================="
echo "✓ System identification complete!"
echo ""
echo "Generated files:"
echo "  - $LATEST_DATA"
echo "  - $MODEL_FILE"
echo "  - $DATA_DIR/data_overview.png"
echo "  - $DATA_DIR/trajectory.png"
echo "  - $DATA_DIR/control_commands.png"
echo "  - $DATA_DIR/training_curves.png"
echo "  - $DATA_DIR/predictions_vs_ground_truth.png"
echo ""
echo "Next steps:"
echo "  - Review the plots to verify model quality"
echo "  - Use the model in your MPC: predictor.py"
echo "=================================================="
