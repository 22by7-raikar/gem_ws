#!/usr/bin/env python3

"""
Test collected data and trained model
Performs validation analysis and visualization
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


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


class DataValidator:
    """Validate collected data and model predictions"""
    
    def __init__(self, data_file, model_file=None, device=None):
        """Initialize validator"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.data_file = Path(data_file).resolve()
        self.model_file = Path(model_file) if model_file else None
        
        # Find workspace root
        data_path = self.data_file.parent
        if data_path.name == 'raw':
            self.workspace_root = data_path.parent.parent
        else:
            self.workspace_root = data_path.parent
        
        print(f"Using device: {self.device}")
        print(f"Workspace root: {self.workspace_root}\n")
        
        # Load data
        self.df = pd.read_csv(data_file)
        print(f"Loaded data: {len(self.df)} samples")
        
        # Load model if provided
        self.model = None
        self.normalizers = None
        if model_file:
            self.load_model(model_file)
    
    def load_model(self, model_file):
        """Load trained model"""
        print(f"\nLoading model: {model_file}")
        checkpoint = torch.load(model_file, map_location=self.device)
        
        config = checkpoint['model_config']
        self.model = VehicleDynamicsNet(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            output_size=config['output_size']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.normalizers = checkpoint['normalizers']
        print("✓ Model loaded\n")
    
    def analyze_data_statistics(self):
        """Analyze and print data statistics"""
        print("="*70)
        print("DATA STATISTICS")
        print("="*70)
        
        state_cols = ['x', 'y', 'yaw', 'vx', 'vy', 'yaw_rate']
        control_cols = ['steering_angle', 'speed']
        
        print("\nState Variables:")
        print("-"*70)
        for col in state_cols:
            data = self.df[col]
            print(f"  {col:15s}: min={data.min():10.4f}, max={data.max():10.4f}, "
                  f"mean={data.mean():10.4f}, std={data.std():10.4f}")
        
        print("\nControl Variables:")
        print("-"*70)
        for col in control_cols:
            data = self.df[col]
            print(f"  {col:15s}: min={data.min():10.4f}, max={data.max():10.4f}, "
                  f"mean={data.mean():10.4f}, std={data.std():10.4f}")
        
        print("\n")
    
    def plot_data_overview(self, output_dir=None):
        """Generate overview plots of collected data"""
        if output_dir is None:
            output_dir = self.workspace_root / "data" / "plots"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Collected Data Overview', fontsize=14, fontweight='bold')
        
        state_cols = ['x', 'y', 'yaw', 'vx', 'vy', 'yaw_rate']
        
        for idx, col in enumerate(state_cols):
            ax = axes[idx // 2, idx % 2]
            ax.plot(self.df[col], linewidth=0.8)
            ax.set_xlabel('Sample')
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / "data_overview.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_control_commands(self, output_dir=None):
        """Plot control input commands over time"""
        if output_dir is None:
            output_dir = self.workspace_root / "data" / "plots"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Control Commands', fontsize=14, fontweight='bold')
        
        control_cols = ['steering_angle', 'speed']
        
        for idx, col in enumerate(control_cols):
            ax = axes[idx]
            ax.plot(self.df[col], linewidth=0.8, label=col)
            ax.set_xlabel('Sample')
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        output_path = output_dir / "control_commands.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_trajectory(self, output_dir=None):
        """Plot vehicle trajectory (x, y)"""
        if output_dir is None:
            output_dir = self.workspace_root / "data" / "plots"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x = self.df['x'].values
        y = self.df['y'].values
        
        # Plot trajectory with color gradient
        scatter = ax.scatter(x, y, c=range(len(x)), cmap='viridis', s=10, alpha=0.6)
        ax.plot(x, y, 'gray', linewidth=0.5, alpha=0.3)
        
        # Mark start and end
        ax.plot(x[0], y[0], 'go', markersize=10, label='Start')
        ax.plot(x[-1], y[-1], 'r*', markersize=15, label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Vehicle Trajectory')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time →')
        ax.legend()
        
        plt.tight_layout()
        output_path = output_dir / "trajectory.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def evaluate_model(self):
        """Evaluate model on test data"""
        if self.model is None or self.normalizers is None:
            print("Model not loaded. Skipping evaluation.")
            return
        
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        state_cols = ['x', 'y', 'yaw', 'vx', 'vy', 'yaw_rate']
        control_cols = ['steering_angle', 'speed']
        
        states = self.df[state_cols].values.astype(np.float32)
        controls = self.df[control_cols].values.astype(np.float32)
        
        # Prepare data
        X_list, y_list = [], []
        for i in range(len(self.df) - 1):
            x = np.concatenate([states[i], controls[i]])
            X_list.append(x)
            y_list.append(states[i + 1])
        
        X = np.array(X_list)
        y_true = np.array(y_list)
        
        # Normalize
        X_norm = (X - self.normalizers['X_mean']) / self.normalizers['X_std']
        y_true_norm = (y_true - self.normalizers['y_mean']) / self.normalizers['y_std']
        
        # Predict
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_norm).float().to(self.device)
            y_pred_norm = self.model(X_tensor).cpu().numpy()
        
        # Denormalize predictions
        y_pred = y_pred_norm * self.normalizers['y_std'] + self.normalizers['y_mean']
        
        # Compute errors
        errors = np.abs(y_pred - y_true)
        mse = np.mean((y_pred - y_true) ** 2)
        rmse = np.sqrt(mse)
        
        print(f"\nMean Squared Error (MSE): {mse:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        
        print("\nPer-state errors:")
        print("-"*70)
        for idx, col in enumerate(state_cols):
            mean_error = np.mean(errors[:, idx])
            max_error = np.max(errors[:, idx])
            print(f"  {col:15s}: mean error={mean_error:.6f}, max error={max_error:.6f}")
        
        # Plot predictions vs ground truth
        self.plot_predictions(y_true, y_pred)
    
    def plot_predictions(self, y_true, y_pred, output_dir=None):
        """Plot model predictions vs ground truth"""
        if output_dir is None:
            output_dir = self.workspace_root / "data" / "plots"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        state_cols = ['x', 'y', 'yaw', 'vx', 'vy', 'yaw_rate']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Model Predictions vs Ground Truth (first 500 samples)', 
                     fontsize=14, fontweight='bold')
        
        max_samples = min(500, len(y_true))
        
        for idx, col in enumerate(state_cols):
            ax = axes[idx // 3, idx % 3]
            ax.plot(y_true[:max_samples, idx], label='Ground Truth', linewidth=1.5)
            ax.plot(y_pred[:max_samples, idx], label='Prediction', linewidth=1.5, alpha=0.7)
            ax.set_xlabel('Sample')
            ax.set_ylabel(col)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / "predictions_vs_ground_truth.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Validate collected data and model')
    parser.add_argument('data_file', type=str, help='Path to CSV data file')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model (optional)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create validator
    validator = DataValidator(args.data_file, args.model)
    
    # Analyze data
    validator.analyze_data_statistics()
    
    # Generate plots
    output_dir = args.output_dir or Path(args.data_file).parent
    validator.plot_data_overview(output_dir)
    validator.plot_control_commands(output_dir)
    validator.plot_trajectory(output_dir)
    
    # Evaluate model if provided
    validator.evaluate_model()
    
    print("\n" + "="*70)
    print("✓ Validation complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
