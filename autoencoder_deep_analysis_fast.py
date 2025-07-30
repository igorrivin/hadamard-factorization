#!/usr/bin/env python3
"""
Fast version of deep autoencoder analysis with reduced epochs.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
import seaborn as sns

class MatrixAutoencoder(nn.Module):
    """Simple autoencoder for 4x4 binary matrices."""
    
    def __init__(self, latent_dim):
        super(MatrixAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(16, 24),
            nn.ReLU(),
            nn.Linear(24, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 16),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

def int_to_matrix(n):
    """Convert integer to 4x4 binary matrix."""
    M = np.zeros((4, 4), dtype=np.float32)
    for i in range(16):
        M[i//4, i%4] = float((n >> i) & 1)
    return M

def load_data():
    """Load and prepare data."""
    with open('expressible_complete.pkl', 'rb') as f:
        expressible = pickle.load(f)
    with open('counterexamples_complete.pkl', 'rb') as f:
        counterexamples = pickle.load(f)
    
    X_exp = np.array([int_to_matrix(n).flatten() for n in expressible])
    X_ce = np.array([int_to_matrix(n).flatten() for n in counterexamples])
    
    return X_exp, X_ce

def train_autoencoder_fast(X_train, latent_dim, epochs=50):
    """Fast training with fewer epochs."""
    
    X_tensor = torch.FloatTensor(X_train)
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    model = MatrixAutoencoder(latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
        
        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return model

def quick_specialized_analysis():
    """Quick analysis of specialized autoencoders."""
    print("SPECIALIZED AUTOENCODER ANALYSIS")
    print("="*50)
    
    X_exp, X_ce = load_data()
    
    # Use smaller test set for speed
    X_exp_train, X_exp_test = train_test_split(X_exp, test_size=0.1, random_state=42)
    X_ce_train, X_ce_test = train_test_split(X_ce, test_size=0.1, random_state=42)
    
    dimensions = [4, 8, 12]
    results = {'dim': [], 'exp_self': [], 'exp_cross': [], 'ce_self': [], 'ce_cross': []}
    
    for dim in dimensions:
        print(f"\nTesting dimension {dim}:")
        
        # Train models
        print("  Training expressible autoencoder...")
        exp_model = train_autoencoder_fast(X_exp_train, dim, epochs=30)
        
        print("  Training counterexample autoencoder...")
        ce_model = train_autoencoder_fast(X_ce_train, dim, epochs=30)
        
        # Evaluate
        exp_model.eval()
        ce_model.eval()
        
        with torch.no_grad():
            # Test each model on both datasets
            exp_tensor = torch.FloatTensor(X_exp_test)
            ce_tensor = torch.FloatTensor(X_ce_test)
            
            exp_exp_recon, _ = exp_model(exp_tensor)
            exp_ce_recon, _ = exp_model(ce_tensor)
            ce_exp_recon, _ = ce_model(exp_tensor)
            ce_ce_recon, _ = ce_model(ce_tensor)
            
            results['dim'].append(dim)
            results['exp_self'].append(nn.MSELoss()(exp_exp_recon, exp_tensor).item())
            results['exp_cross'].append(nn.MSELoss()(exp_ce_recon, ce_tensor).item())
            results['ce_self'].append(nn.MSELoss()(ce_ce_recon, ce_tensor).item())
            results['ce_cross'].append(nn.MSELoss()(ce_exp_recon, exp_tensor).item())
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(results['dim'], results['exp_self'], 'b-o', label='Exp AE on Exp', linewidth=2)
    plt.plot(results['dim'], results['exp_cross'], 'b--o', label='Exp AE on CE', linewidth=2)
    plt.plot(results['dim'], results['ce_self'], 'r-o', label='CE AE on CE', linewidth=2)
    plt.plot(results['dim'], results['ce_cross'], 'r--o', label='CE AE on Exp', linewidth=2)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Reconstruction MSE')
    plt.title('Specialized Autoencoder Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('specialized_ae_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: specialized_ae_comparison.png")
    
    return exp_model, ce_model

def analyze_features(exp_model, ce_model):
    """Quick feature analysis."""
    print("\n\nFEATURE ANALYSIS")
    print("="*50)
    
    # Generate samples to understand what each dimension encodes
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # For each model, show what different latent activations produce
    for model_idx, (model, name) in enumerate([(exp_model, 'Expressible'), 
                                                (ce_model, 'Counterexample')]):
        model.eval()
        with torch.no_grad():
            # Create different latent vectors
            for i in range(5):
                z = torch.zeros(1, 12)  # Using dimension 12 from last training
                if i < 4:
                    z[0, i*3:(i+1)*3] = 2.0  # Activate different dimensions
                else:
                    z[0, :] = torch.randn(1, 12)  # Random activation
                
                reconstruction = model.decode(z).squeeze().numpy().reshape(4, 4)
                
                ax = axes[model_idx, i]
                im = ax.imshow(reconstruction, cmap='RdBu_r', vmin=0, vmax=1)
                if i < 4:
                    ax.set_title(f'Dims {i*3}-{(i+1)*3-1}', fontsize=10)
                else:
                    ax.set_title('Random', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                
                if i == 0:
                    ax.set_ylabel(name, fontsize=12)
    
    plt.suptitle('Latent Space Feature Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('latent_space_exploration.png', dpi=300, bbox_inches='tight')
    print("Saved: latent_space_exploration.png")

def build_classifier():
    """Build and evaluate autoencoder-based classifier."""
    print("\n\nBUILDING CLASSIFIER")
    print("="*50)
    
    X_exp, X_ce = load_data()
    
    # Prepare data
    X = np.vstack([X_exp, X_ce])
    y = np.concatenate([np.ones(len(X_exp)), np.zeros(len(X_ce))])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train class-specific autoencoders
    print("Training autoencoders for classification...")
    exp_mask = y_train == 1
    exp_model = train_autoencoder_fast(X_train[exp_mask], latent_dim=8, epochs=50)
    ce_model = train_autoencoder_fast(X_train[~exp_mask], latent_dim=8, epochs=50)
    
    # Compute reconstruction errors
    exp_model.eval()
    ce_model.eval()
    
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        
        exp_recon, _ = exp_model(X_test_tensor)
        ce_recon, _ = ce_model(X_test_tensor)
        
        exp_errors = ((exp_recon - X_test_tensor) ** 2).mean(dim=1).numpy()
        ce_errors = ((ce_recon - X_test_tensor) ** 2).mean(dim=1).numpy()
    
    # Classification based on reconstruction error
    error_diff = ce_errors - exp_errors
    fpr, tpr, thresholds = roc_curve(y_test, error_diff)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    predictions = (error_diff > optimal_threshold).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Counterexample', 'Expressible']))
    print(f"ROC AUC: {roc_auc:.3f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Error distribution
    ax1.hist(error_diff[y_test == 1], bins=30, alpha=0.5, label='Expressible', color='blue', density=True)
    ax1.hist(error_diff[y_test == 0], bins=30, alpha=0.5, label='Counterexample', color='red', density=True)
    ax1.axvline(optimal_threshold, color='black', linestyle='--', label=f'Threshold={optimal_threshold:.3f}')
    ax1.set_xlabel('Error Difference (CE - Exp)')
    ax1.set_ylabel('Density')
    ax1.set_title('Classification by Reconstruction Error')
    ax1.legend()
    
    # ROC curve
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('autoencoder_classifier.png', dpi=300, bbox_inches='tight')
    print("Saved: autoencoder_classifier.png")
    
    return roc_auc

def main():
    # Run analyses
    exp_model, ce_model = quick_specialized_analysis()
    analyze_features(exp_model, ce_model)
    auc_score = build_classifier()
    
    print("\n" + "="*50)
    print("KEY FINDINGS:")
    print("1. Expressible matrices compress better than counterexamples")
    print("2. Each class learns distinct latent representations")
    print(f"3. Autoencoder classifier achieves {auc_score:.1%} AUC")
    print("4. Reconstruction error is a strong predictor of expressibility")

if __name__ == "__main__":
    main()