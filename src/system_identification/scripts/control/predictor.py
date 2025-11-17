#!/usr/bin/env python3

"""
Model Predictor - Use trained model for vehicle dynamics prediction
Loads trained model and provides inference interface
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse


class VehicleDynamicsNet(nn.Module):
    """Neural network for vehicle dynamics prediction"""
    def __init__(self, input_size=8, hidden_sizes=[128, 128, 64], output_size=6):
        super(VehicleDynamicsNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class ModelPredictor:
    """Predictor class for vehicle dynamics"""
    
    def __init__(self, model_path, device=None):
        """
        Load trained model
        
        Args:
            model_path: Path to saved model checkpoint
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model_path = Path(model_path)
        
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Reconstruct model
        config = checkpoint['model_config']
        self.model = VehicleDynamicsNet(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            output_size=config['output_size']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Store normalizers
        self.normalizers = checkpoint['normalizers']
        self.X_mean = self.normalizers['X_mean']
        self.X_std = self.normalizers['X_std']
        self.y_mean = self.normalizers['y_mean']
        self.y_std = self.normalizers['y_std']
        
        print("✓ Model loaded successfully\n")
    
    def predict_next_state(self, current_state, control_input, return_numpy=True):
        """
        Predict next state given current state and control input
        
        Args:
            current_state: [x, y, yaw, vx, vy, yaw_rate] or numpy array
            control_input: [steering_angle, speed] or numpy array
            return_numpy: If True, return numpy array; if False, return tensor
        
        Returns:
            next_state: Predicted next state [x, y, yaw, vx, vy, yaw_rate]
        """
        # Ensure numpy arrays
        if isinstance(current_state, list):
            current_state = np.array(current_state, dtype=np.float32)
        if isinstance(control_input, list):
            control_input = np.array(control_input, dtype=np.float32)
        
        # Concatenate state and control
        x = np.concatenate([current_state, control_input])
        
        # Normalize
        x_norm = (x - self.X_mean) / self.X_std
        
        # Predict
        with torch.no_grad():
            x_tensor = torch.from_numpy(x_norm).float().unsqueeze(0).to(self.device)
            y_norm_tensor = self.model(x_tensor)
            y_norm = y_norm_tensor.cpu().numpy()[0]
        
        # Denormalize
        next_state = y_norm * self.y_std + self.y_mean
        
        if return_numpy:
            return next_state
        else:
            return torch.from_numpy(next_state).to(self.device)
    
    def predict_trajectory(self, initial_state, control_sequence):
        """
        Predict trajectory given initial state and sequence of controls
        
        Args:
            initial_state: [x, y, yaw, vx, vy, yaw_rate]
            control_sequence: List of [steering_angle, speed] or (N, 2) array
        
        Returns:
            trajectory: (N, 6) array of predicted states
        """
        trajectory = [np.array(initial_state, dtype=np.float32)]
        current_state = np.array(initial_state, dtype=np.float32)
        
        if isinstance(control_sequence, list):
            control_sequence = np.array(control_sequence, dtype=np.float32)
        
        for i, control in enumerate(control_sequence):
            next_state = self.predict_next_state(current_state, control, return_numpy=True)
            trajectory.append(next_state)
            current_state = next_state
        
        return np.array(trajectory)
    
    def predict_steps(self, initial_state, control_input, num_steps=10):
        """
        Repeatedly predict using the same control input for multiple steps
        
        Args:
            initial_state: [x, y, yaw, vx, vy, yaw_rate]
            control_input: [steering_angle, speed]
            num_steps: Number of prediction steps
        
        Returns:
            trajectory: (num_steps+1, 6) array
        """
        trajectory = [np.array(initial_state, dtype=np.float32)]
        current_state = np.array(initial_state, dtype=np.float32)
        
        for _ in range(num_steps):
            next_state = self.predict_next_state(current_state, control_input, return_numpy=True)
            trajectory.append(next_state)
            current_state = next_state
        
        return np.array(trajectory)


def main():
    parser = argparse.ArgumentParser(description='Vehicle dynamics predictor')
    parser.add_argument('model_path', type=str, help='Path to trained model')
    parser.add_argument('--state', type=float, nargs=6, default=[0, 0, 0, 1, 0, 0],
                        help='Initial state [x, y, yaw, vx, vy, yaw_rate]')
    parser.add_argument('--control', type=float, nargs=2, default=[0.1, 1.0],
                        help='Control input [steering_angle, speed]')
    parser.add_argument('--steps', type=int, default=10, help='Number of prediction steps')
    
    args = parser.parse_args()
    
    # Load model
    predictor = ModelPredictor(args.model_path)
    
    # Example 1: Single step prediction
    print("="*60)
    print("Single-step prediction:")
    print("="*60)
    print(f"Initial state:  {args.state}")
    print(f"Control input:  {args.control}")
    
    next_state = predictor.predict_next_state(args.state, args.control)
    print(f"Next state:     {next_state}\n")
    
    # Example 2: Multi-step prediction with constant control
    print("="*60)
    print(f"Multi-step prediction ({args.steps} steps):")
    print("="*60)
    trajectory = predictor.predict_steps(args.state, args.control, num_steps=args.steps)
    
    print("Step | State")
    print("-"*80)
    print(f"  0  | {trajectory[0]}")
    for i in range(1, len(trajectory)):
        print(f"  {i:2d}  | {trajectory[i]}")
    
    print("\n" + "="*60)
    print("Position evolution:")
    print("="*60)
    for i, state in enumerate(trajectory):
        print(f"Step {i:2d}: x={state[0]:8.3f}, y={state[1]:8.3f}, yaw={state[2]:7.3f}")


if __name__ == '__main__':
    main()
