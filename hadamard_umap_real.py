#!/usr/bin/env python3
"""
UMAP visualization of Hadamard factorizability over the reals.
Color matrices by factorizability and explore patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our solver
from hadamard_real_demo import solve_hadamard_alternating

def generate_matrix_with_pattern(zero_pattern, scale=1.0):
    """
    Generate a random 4x4 matrix with specified zero pattern.
    zero_pattern is a 16-bit integer where bit i indicates if position i is zero.
    """
    matrix = np.random.randn(4, 4) * scale
    
    # Apply zero pattern
    for i in range(16):
        if zero_pattern & (1 << i):
            row, col = i // 4, i % 4
            matrix[row, col] = 0.0
    
    return matrix

def test_factorizability(C, n_trials=5, tol=1e-8):
    """
    Test if matrix C is factorizable as Hadamard product of rank-2 matrices.
    Returns (is_factorizable, best_error, best_rank_A, best_rank_B)
    """
    best_error = float('inf')
    best_ranks = (None, None)
    
    for trial in range(n_trials):
        try:
            A, B, error, _ = solve_hadamard_alternating(C, max_iter=200, tol=tol)
            
            if error < best_error:
                best_error = error
                best_ranks = (np.linalg.matrix_rank(A), np.linalg.matrix_rank(B))
                
        except:
            continue
    
    rel_error = best_error / (np.linalg.norm(C, 'fro') + 1e-10)
    is_factorizable = rel_error < 1e-6 and best_ranks[0] <= 2 and best_ranks[1] <= 2
    
    return is_factorizable, rel_error, best_ranks

def compute_matrix_features(C):
    """
    Compute feature vector for a matrix.
    Includes: entries, singular values, zero pattern info, symmetry measures.
    """
    features = []
    
    # Raw entries (16 features)
    features.extend(C.flatten())
    
    # Singular values (4 features)
    s = np.linalg.svd(C, compute_uv=False)
    features.extend(s)
    
    # Zero pattern statistics (5 features)
    n_zeros = np.sum(np.abs(C) < 1e-10)
    features.append(n_zeros)
    features.append(np.sum(np.abs(C[0, :]) < 1e-10))  # zeros in first row
    features.append(np.sum(np.abs(C[:, 0]) < 1e-10))  # zeros in first col
    features.append(np.sum(np.abs(np.diag(C)) < 1e-10))  # zeros on diagonal
    features.append(np.sum(np.abs(C[np.triu_indices(4, 1)]) < 1e-10))  # zeros in upper triangle
    
    # Matrix properties (6 features)
    features.append(np.linalg.norm(C, 'fro'))
    features.append(np.linalg.norm(C, 1))
    features.append(np.linalg.norm(C, np.inf))
    # Clip condition number to avoid infinity
    cond = np.linalg.cond(C)
    features.append(np.log10(cond + 1) if cond < 1e10 else 10.0)
    features.append(np.linalg.matrix_rank(C))
    # Clip determinant to reasonable range
    det = np.linalg.det(C)
    features.append(np.clip(det, -1000, 1000))
    
    # Symmetry measures (3 features)
    features.append(np.linalg.norm(C - C.T, 'fro'))  # asymmetry
    features.append(np.linalg.norm(C + C.T, 'fro'))  # symmetric part
    features.append(np.linalg.norm(C - np.diag(np.diag(C)), 'fro'))  # off-diagonal norm
    
    return np.array(features)

def generate_dataset(n_samples=2000, n_zeros_range=(0, 10)):
    """
    Generate dataset of matrices with various zero patterns.
    """
    print("Generating dataset of random matrices...")
    
    matrices = []
    features = []
    labels = []  # 0: not factorizable, 1: factorizable
    metadata = []
    
    # Sample across different numbers of zeros
    samples_per_zero_count = n_samples // (n_zeros_range[1] - n_zeros_range[0] + 1)
    
    pbar = tqdm(total=n_samples, desc="Testing matrices")
    
    for n_zeros in range(n_zeros_range[0], n_zeros_range[1] + 1):
        for _ in range(samples_per_zero_count):
            # Generate random zero pattern with n_zeros
            if n_zeros == 0:
                zero_pattern = 0
            elif n_zeros == 16:
                zero_pattern = (1 << 16) - 1
            else:
                # Random pattern with exactly n_zeros
                positions = np.random.choice(16, n_zeros, replace=False)
                zero_pattern = sum(1 << p for p in positions)
            
            # Generate matrix
            C = generate_matrix_with_pattern(zero_pattern)
            
            # Test factorizability
            is_fact, error, ranks = test_factorizability(C)
            
            # Store results
            matrices.append(C)
            features.append(compute_matrix_features(C))
            labels.append(int(is_fact))
            metadata.append({
                'n_zeros': n_zeros,
                'zero_pattern': zero_pattern,
                'error': error,
                'ranks': ranks,
                'matrix_rank': np.linalg.matrix_rank(C)
            })
            
            pbar.update(1)
    
    pbar.close()
    
    # Convert to arrays
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"\nDataset summary:")
    print(f"Total matrices: {len(matrices)}")
    print(f"Factorizable: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
    
    return matrices, features, labels, metadata

def create_umap_visualization(features, labels, metadata):
    """
    Create UMAP visualization colored by factorizability.
    """
    print("\nComputing UMAP embedding...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Compute UMAP
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(features_scaled)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('UMAP Analysis of Hadamard Factorizability (Real Matrices)', fontsize=16)
    
    # 1. Basic factorizability
    ax = axes[0, 0]
    colors = ['red', 'green']
    for i, (label, color) in enumerate(zip([0, 1], colors)):
        mask = labels == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                  c=color, alpha=0.6, s=20,
                  label=f'{"Factorizable" if label else "Not factorizable"}')
    ax.set_title('Factorizability')
    ax.legend()
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # 2. Color by number of zeros
    ax = axes[0, 1]
    n_zeros = np.array([m['n_zeros'] for m in metadata])
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                        c=n_zeros, cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=ax, label='Number of zeros')
    ax.set_title('Number of Zeros')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # 3. Color by matrix rank
    ax = axes[0, 2]
    ranks = np.array([m['matrix_rank'] for m in metadata])
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                        c=ranks, cmap='coolwarm', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=ax, label='Matrix rank')
    ax.set_title('Matrix Rank')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # 4. Factorizability by zeros (heatmap style)
    ax = axes[1, 0]
    # Create custom colormap
    colors_custom = ['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'green']
    n_bins = 6
    cmap = ListedColormap(colors_custom)
    
    # Compute factorizability score (0-1) with number of zeros
    fact_score = np.zeros(len(labels))
    for i in range(len(labels)):
        fact_score[i] = labels[i] + 0.5 * (10 - n_zeros[i]) / 10
    
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                        c=fact_score, cmap=cmap, alpha=0.7, s=30)
    plt.colorbar(scatter, ax=ax, label='Factorizability + density')
    ax.set_title('Combined: Factorizability and Zero Density')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # 5. Error magnitude for factorizable matrices
    ax = axes[1, 1]
    errors = np.array([m['error'] for m in metadata])
    log_errors = np.log10(errors + 1e-16)
    
    # Only show factorizable ones
    mask = labels == 1
    scatter = ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                        c=log_errors[mask], cmap='plasma', alpha=0.7, s=30)
    plt.colorbar(scatter, ax=ax, label='log10(relative error)')
    ax.set_title('Approximation Quality (Factorizable Only)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # 6. Zero pattern clusters
    ax = axes[1, 2]
    # Group by zero pattern
    unique_patterns = {}
    pattern_colors = {}
    for i, m in enumerate(metadata):
        pattern = m['zero_pattern']
        if pattern not in unique_patterns:
            unique_patterns[pattern] = len(unique_patterns)
            pattern_colors[pattern] = plt.cm.tab20(len(unique_patterns) % 20)
    
    colors_array = [pattern_colors[m['zero_pattern']] for m in metadata]
    
    for pattern, idx in unique_patterns.items():
        mask = [m['zero_pattern'] == pattern for m in metadata]
        if sum(mask) > 5:  # Only show patterns with enough samples
            ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                      c=[colors_array[i] for i, m in enumerate(mask) if m],
                      alpha=0.6, s=20)
    
    ax.set_title('Zero Pattern Clusters')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    plt.tight_layout()
    plt.savefig('hadamard_umap_real.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to hadamard_umap_real.png")
    
    return embedding

def analyze_clusters(embedding, labels, metadata):
    """
    Analyze what distinguishes factorizable from non-factorizable matrices.
    """
    print("\n" + "="*60)
    print("CLUSTER ANALYSIS")
    print("="*60)
    
    # Factorizability by number of zeros
    print("\nFactorizability rate by number of zeros:")
    n_zeros_list = sorted(set(m['n_zeros'] for m in metadata))
    
    for n_zeros in n_zeros_list:
        mask = [m['n_zeros'] == n_zeros for m in metadata]
        subset_labels = labels[mask]
        if len(subset_labels) > 0:
            rate = np.mean(subset_labels) * 100
            print(f"  {n_zeros:2d} zeros: {rate:5.1f}% factorizable ({sum(mask)} samples)")
    
    # Factorizability by rank
    print("\nFactorizability rate by matrix rank:")
    for rank in [1, 2, 3, 4]:
        mask = [m['matrix_rank'] == rank for m in metadata]
        subset_labels = labels[mask]
        if len(subset_labels) > 0:
            rate = np.mean(subset_labels) * 100
            print(f"  Rank {rank}: {rate:5.1f}% factorizable ({sum(mask)} samples)")
    
    # Find boundary between factorizable and non-factorizable
    print("\nBoundary analysis:")
    from sklearn.neighbors import NearestNeighbors
    
    # Find mixed neighborhoods
    nbrs = NearestNeighbors(n_neighbors=10)
    nbrs.fit(embedding)
    
    boundary_score = []
    for i in range(len(labels)):
        _, indices = nbrs.kneighbors([embedding[i]])
        neighbor_labels = labels[indices[0]]
        # High score if neighbors have different labels
        score = np.std(neighbor_labels)
        boundary_score.append(score)
    
    boundary_score = np.array(boundary_score)
    boundary_indices = np.argsort(boundary_score)[-20:]  # Top 20 boundary points
    
    print(f"\nBoundary matrices (hardest cases):")
    for idx in boundary_indices[:5]:
        m = metadata[idx]
        print(f"  Matrix {idx}: {m['n_zeros']} zeros, rank {m['matrix_rank']}, "
              f"{'factorizable' if labels[idx] else 'not factorizable'}, "
              f"error={m['error']:.2e}")

if __name__ == "__main__":
    # Generate dataset
    matrices, features, labels, metadata = generate_dataset(
        n_samples=2000,
        n_zeros_range=(0, 12)  # Focus on interesting range
    )
    
    # Create visualization
    embedding = create_umap_visualization(features, labels, metadata)
    
    # Analyze clusters
    analyze_clusters(embedding, labels, metadata)
    
    # Additional plot: Success rate curve
    plt.figure(figsize=(10, 6))
    n_zeros_list = sorted(set(m['n_zeros'] for m in metadata))
    success_rates = []
    
    for n_zeros in n_zeros_list:
        mask = [m['n_zeros'] == n_zeros for m in metadata]
        rate = np.mean(labels[mask]) * 100 if sum(mask) > 0 else 0
        success_rates.append(rate)
    
    plt.plot(n_zeros_list, success_rates, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of zeros', fontsize=12)
    plt.ylabel('Factorization success rate (%)', fontsize=12)
    plt.title('Hadamard Factorizability vs Matrix Sparsity (Real Numbers)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 105)
    
    # Add comparison lines
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    plt.text(max(n_zeros_list)*0.7, 52, '50%', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig('hadamard_success_rate_real.png', dpi=300, bbox_inches='tight')
    print("\nSaved success rate plot to hadamard_success_rate_real.png")
    
    print("\n✓ Analysis complete!")