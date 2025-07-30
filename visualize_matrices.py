#!/usr/bin/env python3
"""
Visualize the landscape of expressible and non-expressible matrices using UMAP.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import seaborn as sns
from collections import defaultdict

def int_to_matrix(n):
    """Convert 16-bit integer to 4×4 binary matrix."""
    M = np.zeros((4, 4), dtype=int)
    for i in range(16):
        M[i//4, i%4] = (n >> i) & 1
    return M

def compute_features(matrix_list):
    """Compute various features for each matrix."""
    features = []
    
    for n in matrix_list:
        M = int_to_matrix(n)
        
        # Basic features
        flat = M.flatten()
        
        # Additional features
        row_sums = M.sum(axis=1)
        col_sums = M.sum(axis=0)
        diag_sum = np.trace(M)
        antidiag_sum = np.trace(np.fliplr(M))
        
        # Symmetry features
        is_symmetric = int(np.allclose(M, M.T))
        is_antisymmetric = int(np.allclose(M, -M.T))
        
        # Block structure
        block_00 = M[:2, :2].sum()
        block_01 = M[:2, 2:].sum()
        block_10 = M[2:, :2].sum()
        block_11 = M[2:, 2:].sum()
        
        # Combine all features
        feature_vec = np.concatenate([
            flat,  # 16 features
            row_sums,  # 4 features
            col_sums,  # 4 features
            [diag_sum, antidiag_sum, is_symmetric, is_antisymmetric],  # 4 features
            [block_00, block_01, block_10, block_11]  # 4 features
        ])
        
        features.append(feature_vec)
    
    return np.array(features)

def plot_umap_results(embedding, labels, title="UMAP Visualization"):
    """Create a visualization of UMAP results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Colored by expressibility
    colors = ['red' if l == 0 else 'blue' for l in labels]
    ax1.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.5, s=10)
    ax1.set_title(f"{title} - Colored by Expressibility")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    
    # Add legend
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Counterexamples')
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='Expressible')
    ax1.legend(handles=[red_patch, blue_patch])
    
    # Plot 2: Density plot
    ax2.hexbin(embedding[:, 0], embedding[:, 1], gridsize=50, cmap='YlOrRd')
    ax2.set_title(f"{title} - Density Plot")
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    
    plt.tight_layout()
    return fig

def analyze_clusters(embedding, labels, matrix_list):
    """Analyze what makes different clusters special."""
    from sklearn.cluster import DBSCAN
    
    # Find clusters in UMAP space
    clustering = DBSCAN(eps=0.5, min_samples=10).fit(embedding)
    cluster_labels = clustering.labels_
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"\nFound {n_clusters} clusters in UMAP space")
    
    # Analyze each cluster
    cluster_info = defaultdict(lambda: {'expressible': 0, 'counterexample': 0, 'matrices': []})
    
    for i, (cluster, is_expressible) in enumerate(zip(cluster_labels, labels)):
        if cluster != -1:  # Ignore noise points
            if is_expressible:
                cluster_info[cluster]['expressible'] += 1
            else:
                cluster_info[cluster]['counterexample'] += 1
            cluster_info[cluster]['matrices'].append(matrix_list[i])
    
    # Report on clusters
    print("\nCluster Analysis:")
    for cluster_id in sorted(cluster_info.keys()):
        info = cluster_info[cluster_id]
        total = info['expressible'] + info['counterexample']
        exp_pct = info['expressible'] / total * 100
        print(f"\nCluster {cluster_id}: {total} matrices")
        print(f"  Expressible: {info['expressible']} ({exp_pct:.1f}%)")
        print(f"  Counterexamples: {info['counterexample']} ({100-exp_pct:.1f}%)")
        
        # Show example from this cluster
        example = int_to_matrix(info['matrices'][0])
        print(f"  Example matrix:")
        print(f"  {example}")

def create_zero_pattern_visualization(expressible, counterexamples):
    """Visualize based on zero patterns."""
    all_matrices = list(expressible) + list(counterexamples)
    labels = [1] * len(expressible) + [0] * len(counterexamples)
    
    # Create feature matrix based on zero positions
    features = []
    for n in all_matrices:
        M = int_to_matrix(n)
        # Use binary encoding of zero positions
        zero_pattern = (M == 0).flatten().astype(int)
        features.append(zero_pattern)
    
    features = np.array(features)
    
    # Apply UMAP
    print("Running UMAP on zero patterns...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(features)
    
    return embedding, labels, all_matrices

def main():
    print("VISUALIZING HADAMARD FACTORIZATION LANDSCAPE")
    print("="*50)
    
    # Load data
    try:
        with open('counterexamples_complete.pkl', 'rb') as f:
            counterexamples = pickle.load(f)
        with open('expressible_complete.pkl', 'rb') as f:
            expressible = pickle.load(f)
    except FileNotFoundError:
        print("Error: Run hadamard_search.py first to generate data!")
        return
    
    print(f"Loaded {len(counterexamples)} counterexamples")
    print(f"Loaded {len(expressible)} expressible matrices")
    
    # Sample for visualization (UMAP can be slow on large datasets)
    n_samples = min(5000, len(expressible))
    expressible_sample = list(np.random.choice(expressible, n_samples, replace=False))
    counterexample_sample = list(counterexamples)  # Use all counterexamples
    
    # Combine data
    all_matrices = expressible_sample + counterexample_sample
    labels = [1] * len(expressible_sample) + [0] * len(counterexample_sample)
    
    print(f"\nUsing {len(all_matrices)} matrices for visualization")
    
    # Method 1: Full feature representation
    print("\n1. Computing feature representations...")
    features = compute_features(all_matrices)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA first for denoising
    print("   Applying PCA...")
    pca = PCA(n_components=10)
    features_pca = pca.fit_transform(features_scaled)
    print(f"   PCA explained variance: {pca.explained_variance_ratio_.sum():.2f}")
    
    # Apply UMAP
    print("   Applying UMAP...")
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(features_pca)
    
    # Visualize
    fig1 = plot_umap_results(embedding, labels, "Full Feature UMAP")
    plt.savefig('umap_full_features.png', dpi=300, bbox_inches='tight')
    print("   Saved: umap_full_features.png")
    
    # Method 2: Zero pattern only
    print("\n2. Visualizing based on zero patterns only...")
    zero_embedding, zero_labels, zero_matrices = create_zero_pattern_visualization(
        expressible_sample, counterexample_sample
    )
    
    fig2 = plot_umap_results(zero_embedding, zero_labels, "Zero Pattern UMAP")
    plt.savefig('umap_zero_patterns.png', dpi=300, bbox_inches='tight')
    print("   Saved: umap_zero_patterns.png")
    
    # Analyze clusters
    print("\n3. Analyzing clusters in full feature space...")
    analyze_clusters(embedding, labels, all_matrices)
    
    # Create a heatmap of average matrices
    print("\n4. Creating average matrix heatmaps...")
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Average expressible matrix
    avg_expressible = np.zeros((4, 4))
    for n in expressible_sample:
        avg_expressible += int_to_matrix(n)
    avg_expressible /= len(expressible_sample)
    
    # Average counterexample
    avg_counterexample = np.zeros((4, 4))
    for n in counterexample_sample:
        avg_counterexample += int_to_matrix(n)
    avg_counterexample /= len(counterexample_sample)
    
    sns.heatmap(avg_expressible, annot=True, fmt='.3f', cmap='Blues', ax=ax1, vmin=0, vmax=1)
    ax1.set_title('Average Expressible Matrix')
    
    sns.heatmap(avg_counterexample, annot=True, fmt='.3f', cmap='Reds', ax=ax2, vmin=0, vmax=1)
    ax2.set_title('Average Counterexample')
    
    plt.tight_layout()
    plt.savefig('average_matrices.png', dpi=300, bbox_inches='tight')
    print("   Saved: average_matrices.png")
    
    # Distribution of number of ones
    print("\n5. Creating distribution plots...")
    fig4, ax = plt.subplots(figsize=(10, 6))
    
    ones_expressible = [bin(n).count('1') for n in expressible]
    ones_counterexample = [bin(n).count('1') for n in counterexamples]
    
    bins = range(0, 17)
    ax.hist(ones_expressible, bins=bins, alpha=0.5, label='Expressible', density=True, color='blue')
    ax.hist(ones_counterexample, bins=bins, alpha=0.5, label='Counterexamples', density=True, color='red')
    ax.set_xlabel('Number of ones in matrix')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Number of Ones')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ones_distribution.png', dpi=300, bbox_inches='tight')
    print("   Saved: ones_distribution.png")
    
    plt.show()
    
    print("\nVisualization complete!")
    print("Generated files:")
    print("- umap_full_features.png: UMAP based on all features")
    print("- umap_zero_patterns.png: UMAP based on zero patterns only")
    print("- average_matrices.png: Heatmaps of average matrices")
    print("- ones_distribution.png: Distribution of number of ones")

if __name__ == "__main__":
    main()