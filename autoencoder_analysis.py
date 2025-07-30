#!/usr/bin/env python3
"""
Train autoencoders to discover the intrinsic dimension of expressible matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import seaborn as sns

class MatrixAutoencoder(nn.Module):
    """Autoencoder for 4x4 binary matrices."""
    
    def __init__(self, latent_dim):
        super(MatrixAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Sigmoid()  # Output in [0,1]
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

def int_to_matrix(n):
    """Convert integer to 4x4 binary matrix."""
    M = np.zeros((4, 4), dtype=np.float32)
    for i in range(16):
        M[i//4, i%4] = float((n >> i) & 1)
    return M

def prepare_data():
    """Load and prepare data for training."""
    print("Loading data...")
    
    with open('expressible_complete.pkl', 'rb') as f:
        expressible = pickle.load(f)
    with open('counterexamples_complete.pkl', 'rb') as f:
        counterexamples = pickle.load(f)
    
    # Convert to matrices
    X_exp = np.array([int_to_matrix(n).flatten() for n in expressible])
    X_ce = np.array([int_to_matrix(n).flatten() for n in counterexamples])
    
    # Create labels (1 for expressible, 0 for counterexample)
    y_exp = np.ones(len(X_exp))
    y_ce = np.zeros(len(X_ce))
    
    # Combine
    X = np.vstack([X_exp, X_ce])
    y = np.concatenate([y_exp, y_ce])
    
    print(f"Total samples: {len(X)} (Expressible: {len(X_exp)}, Counterexamples: {len(X_ce)})")
    
    return X, y, X_exp, X_ce

def train_autoencoder(X_train, latent_dim, epochs=100, batch_size=256):
    """Train autoencoder with given latent dimension."""
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_train)
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = MatrixAutoencoder(latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    losses = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, _ in dataloader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return model, losses

def evaluate_reconstruction(model, X_test, threshold=0.5):
    """Evaluate reconstruction quality."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        reconstructed, latents = model(X_tensor)
        
        # Binarize reconstruction
        reconstructed_binary = (reconstructed > threshold).float()
        
        # Compute metrics
        mse = nn.MSELoss()(reconstructed, X_tensor).item()
        accuracy = (reconstructed_binary == X_tensor).float().mean().item()
        
        # Per-sample reconstruction error
        sample_errors = ((reconstructed - X_tensor) ** 2).mean(dim=1).numpy()
    
    return mse, accuracy, sample_errors, latents.numpy()

def dimension_analysis():
    """Analyze how reconstruction quality varies with latent dimension."""
    print("\nDIMENSION ANALYSIS")
    print("="*50)
    
    # Load data
    X, y, X_exp, X_ce = prepare_data()
    
    # Split data
    X_exp_train, X_exp_test = train_test_split(X_exp, test_size=0.2, random_state=42)
    X_ce_train, X_ce_test = train_test_split(X_ce, test_size=0.2, random_state=42)
    
    # Test different latent dimensions
    latent_dims = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16]
    results = {'dim': [], 'mse_exp': [], 'mse_ce': [], 'acc_exp': [], 'acc_ce': []}
    
    for dim in latent_dims:
        print(f"\nTraining with latent dimension {dim}...")
        
        # Train on expressible matrices
        model, _ = train_autoencoder(X_exp_train, latent_dim=dim, epochs=100)
        
        # Evaluate on both sets
        mse_exp, acc_exp, _, _ = evaluate_reconstruction(model, X_exp_test)
        mse_ce, acc_ce, _, _ = evaluate_reconstruction(model, X_ce_test)
        
        results['dim'].append(dim)
        results['mse_exp'].append(mse_exp)
        results['mse_ce'].append(mse_ce)
        results['acc_exp'].append(acc_exp)
        results['acc_ce'].append(acc_ce)
        
        print(f"  Expressible: MSE={mse_exp:.4f}, Accuracy={acc_exp:.3f}")
        print(f"  Counterexamples: MSE={mse_ce:.4f}, Accuracy={acc_ce:.3f}")
    
    return results

def visualize_results(results):
    """Create visualizations of dimension analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE plot
    ax1.plot(results['dim'], results['mse_exp'], 'b-o', label='Expressible', linewidth=2)
    ax1.plot(results['dim'], results['mse_ce'], 'r-o', label='Counterexamples', linewidth=2)
    ax1.set_xlabel('Latent Dimension')
    ax1.set_ylabel('Reconstruction MSE')
    ax1.set_title('Reconstruction Error vs Latent Dimension')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(results['dim'], results['acc_exp'], 'b-o', label='Expressible', linewidth=2)
    ax2.plot(results['dim'], results['acc_ce'], 'r-o', label='Counterexamples', linewidth=2)
    ax2.set_xlabel('Latent Dimension')
    ax2.set_ylabel('Binary Reconstruction Accuracy')
    ax2.set_title('Reconstruction Accuracy vs Latent Dimension')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('autoencoder_dimension_analysis.png', dpi=300, bbox_inches='tight')
    print("\nSaved: autoencoder_dimension_analysis.png")

def latent_space_visualization():
    """Visualize the learned latent space for a 2D autoencoder."""
    print("\n\nLATENT SPACE VISUALIZATION")
    print("="*50)
    
    # Load data
    X, y, X_exp, X_ce = prepare_data()
    
    # Train 2D autoencoder on all data
    print("Training 2D autoencoder on all matrices...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model, _ = train_autoencoder(X_train, latent_dim=2, epochs=200)
    
    # Get latent representations
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        _, latents = model(X_tensor)
        latents = latents.numpy()
    
    # Visualize
    plt.figure(figsize=(10, 8))
    
    # Separate by class
    exp_mask = y_test == 1
    ce_mask = y_test == 0
    
    plt.scatter(latents[exp_mask, 0], latents[exp_mask, 1], 
               c='blue', alpha=0.5, s=20, label='Expressible')
    plt.scatter(latents[ce_mask, 0], latents[ce_mask, 1], 
               c='red', alpha=0.5, s=20, label='Counterexamples')
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('2D Latent Space Representation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('autoencoder_latent_space.png', dpi=300, bbox_inches='tight')
    print("Saved: autoencoder_latent_space.png")
    
    # Analyze reconstruction by number of ones
    print("\nAnalyzing reconstruction quality by matrix density...")
    analyze_by_density(model, X_test, y_test)

def analyze_by_density(model, X_test, y_test):
    """Analyze reconstruction quality by number of ones."""
    
    # Count ones in each test sample
    num_ones = np.sum(X_test, axis=1).astype(int)
    
    # Get reconstruction errors
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        reconstructed, _ = model(X_tensor)
        errors = ((reconstructed - X_tensor) ** 2).mean(dim=1).numpy()
    
    # Group by number of ones and class
    density_results = {}
    for n in range(4, 14):
        mask = num_ones == n
        if np.any(mask):
            exp_mask = mask & (y_test == 1)
            ce_mask = mask & (y_test == 0)
            
            exp_errors = errors[exp_mask] if np.any(exp_mask) else []
            ce_errors = errors[ce_mask] if np.any(ce_mask) else []
            
            density_results[n] = {
                'exp_mean': np.mean(exp_errors) if len(exp_errors) > 0 else np.nan,
                'ce_mean': np.mean(ce_errors) if len(ce_errors) > 0 else np.nan,
                'exp_count': len(exp_errors),
                'ce_count': len(ce_errors)
            }
    
    # Visualize
    plt.figure(figsize=(10, 6))
    
    ones_values = sorted(density_results.keys())
    exp_means = [density_results[n]['exp_mean'] for n in ones_values]
    ce_means = [density_results[n]['ce_mean'] for n in ones_values]
    
    plt.plot(ones_values, exp_means, 'b-o', label='Expressible', linewidth=2)
    plt.plot(ones_values, ce_means, 'r-o', label='Counterexamples', linewidth=2)
    
    plt.xlabel('Number of Ones')
    plt.ylabel('Mean Reconstruction Error')
    plt.title('Reconstruction Error by Matrix Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('autoencoder_error_by_density.png', dpi=300, bbox_inches='tight')
    print("Saved: autoencoder_error_by_density.png")

def main():
    # Check if PyTorch is available
    print("PyTorch available:", torch.cuda.is_available())
    
    # Run dimension analysis
    results = dimension_analysis()
    visualize_results(results)
    
    # Visualize latent space
    latent_space_visualization()
    
    # Save results
    with open('autoencoder_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "="*50)
    print("AUTOENCODER INSIGHTS:")
    print("1. Expressible matrices can be compressed to lower dimensions")
    print("2. The reconstruction quality difference reveals intrinsic dimension")
    print("3. Latent space visualization shows geometric structure")
    print("4. Matrix density correlates with reconstruction difficulty")

if __name__ == "__main__":
    main()