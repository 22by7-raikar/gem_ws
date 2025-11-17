#!/usr/bin/env python3

"""
Train vehicle dynamics model using PyTorch
Learns mapping: (state, control) -> next_state
Input: State [x, y, yaw, vx, vy, yaw_rate] + Control [steering_angle, speed]
Output: Next State [x, y, yaw, vx, vy, yaw_rate]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class SystemIdentification:
    def __init__(self, data_path, device=None):
        """Initialize system identification trainer"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        # Find workspace root (parent of 'data' directory)
        data_path = Path(data_path).resolve()
        if data_path.name == 'raw':
            self.workspace_root = data_path.parent.parent
        elif data_path.name == 'data':
            self.workspace_root = data_path.parent
        else:
            self.workspace_root = data_path.parent.parent
        
        print(f"Using device: {self.device}")
        print(f"Workspace root: {self.workspace_root}")
    
    def load_and_prepare_data(self, csv_file, sequence_length=1, train_split=0.8):
        """
        Load data from CSV and prepare sequences
        
        Args:
            csv_file: Path to CSV file
            sequence_length: Number of time steps to use for prediction
            train_split: Fraction of data for training (0-1)
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test, normalizers)
        """
        print(f"\nLoading data from {csv_file}")
        df = pd.read_csv(csv_file)
        
        # State columns: [x, y, yaw, vx, vy, yaw_rate]
        # Control columns: [steering_angle, speed]
        state_cols = ['x', 'y', 'yaw', 'vx', 'vy', 'yaw_rate']
        control_cols = ['steering_angle', 'speed']
        
        states = df[state_cols].values.astype(np.float32)
        controls = df[control_cols].values.astype(np.float32)
        
        print(f"Total samples: {len(df)}")
        print(f"State shape: {states.shape}")
        print(f"Control shape: {controls.shape}")
        
        # Create sequences: [state_t, control_t] -> state_{t+1}
        sequences_x, sequences_y = [], []
        
        for i in range(len(df) - sequence_length):
            # Input: current state + control
            current_state = states[i]
            current_control = controls[i]
            x = np.concatenate([current_state, current_control])
            
            # Target: next state
            next_state = states[i + sequence_length]
            
            sequences_x.append(x)
            sequences_y.append(next_state)
        
        X = np.array(sequences_x)
        y = np.array(sequences_y)
        
        print(f"Created {len(X)} sequences")
        
        # Compute normalization statistics
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        y_mean = y.mean(axis=0)
        y_std = y.std(axis=0) + 1e-8
        
        # Normalize
        X_norm = (X - X_mean) / X_std
        y_norm = (y - y_mean) / y_std
        
        # Train-test split
        split_idx = int(len(X) * train_split)
        X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
        y_train, y_test = y_norm[:split_idx], y_norm[split_idx:]
        
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Convert to tensors
        X_train = torch.from_numpy(X_train).to(self.device)
        y_train = torch.from_numpy(y_train).to(self.device)
        X_test = torch.from_numpy(X_test).to(self.device)
        y_test = torch.from_numpy(y_test).to(self.device)
        
        normalizers = {
            'X_mean': X_mean,
            'X_std': X_std,
            'y_mean': y_mean,
            'y_std': y_std,
        }
        
        return X_train, y_train, X_test, y_test, normalizers
    
    def train(self, csv_file, epochs=100, batch_size=32, lr=0.001, 
              sequence_length=1, train_split=0.8):
        """Train the vehicle dynamics model"""
        
        # Load and prepare data
        X_train, y_train, X_test, y_test, normalizers = self.load_and_prepare_data(
            csv_file, sequence_length=sequence_length, train_split=train_split
        )
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        model = VehicleDynamicsNet(input_size=8, hidden_sizes=[128, 128, 64], output_size=6)
        model = model.to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        
        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Sequence length: {sequence_length}")
        print(f"{'='*60}\n")
        
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test)
                test_loss = criterion(test_pred, y_test).item()
            
            test_losses.append(test_loss)
            scheduler.step()
            
            # Save best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_state = model.state_dict().copy()
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.6f} | "
                      f"Test Loss: {test_loss:.6f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Save model and normalizers
        models_dir = self.workspace_root / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "vehicle_dynamics_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': 8,
                'hidden_sizes': [128, 128, 64],
                'output_size': 6,
            },
            'normalizers': normalizers,
        }, model_path)
        
        print(f"\n{'='*60}")
        print(f"Model saved to: {model_path}")
        print(f"  Best test loss: {best_test_loss:.6f}")
        print(f"{'='*60}\n")
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(test_losses, label='Test Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Vehicle Dynamics Model Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plots_dir = self.workspace_root / "data" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"Training curves saved to: {plot_path}\n")
        plt.close()
        
        return model, normalizers


def main():
    parser = argparse.ArgumentParser(description='Train vehicle dynamics model')
    parser.add_argument('csv_file', type=str, help='Path to CSV file with collected data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--sequence-length', type=int, default=1, help='Sequence length')
    parser.add_argument('--train-split', type=float, default=0.8, help='Train/test split ratio')
    
    args = parser.parse_args()
    
    # Determine data directory from CSV file
    csv_path = Path(args.csv_file)
    data_dir = csv_path.parent
    
    sys_id = SystemIdentification(data_path=data_dir)
    sys_id.train(
        csv_file=args.csv_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sequence_length=args.sequence_length,
        train_split=args.train_split
    )


if __name__ == '__main__':
    main()
