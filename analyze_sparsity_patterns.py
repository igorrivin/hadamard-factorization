#!/usr/bin/env python3
"""
Analyze and visualize factorization patterns for each sparsity level.
Show which specific zero patterns are factorizable within each group.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from hadamard_real_demo import solve_hadamard_alternating

def pattern_to_matrix_viz(pattern, size=4):
    """Convert a zero pattern (integer) to a matrix visualization."""
    matrix = np.ones((size, size))
    for i in range(size*size):
        if pattern & (1 << i):
            matrix[i // size, i % size] = 0
    return matrix

def generate_all_patterns(n_zeros, size=4):
    """Generate all possible patterns with exactly n_zeros."""
    total_positions = size * size
    patterns = []
    
    if n_zeros == 0:
        return [0]
    elif n_zeros == total_positions:
        return [(1 << total_positions) - 1]
    else:
        for positions in combinations(range(total_positions), n_zeros):
            pattern = sum(1 << p for p in positions)
            patterns.append(pattern)
    
    return patterns

def test_pattern_factorizability(pattern, n_samples=10, scale=1.0):
    """Test if matrices with given zero pattern are typically factorizable."""
    successes = 0
    errors = []
    
    for _ in range(n_samples):
        # Generate random matrix with pattern
        matrix = np.random.randn(4, 4) * scale
        for i in range(16):
            if pattern & (1 << i):
                matrix[i // 4, i % 4] = 0
        
        # Test factorizability
        try:
            A, B, error, _ = solve_hadamard_alternating(matrix, max_iter=200)
            rel_error = error / (np.linalg.norm(matrix, 'fro') + 1e-10)
            
            if rel_error < 1e-6:
                successes += 1
            errors.append(rel_error)
        except:
            errors.append(1.0)
    
    success_rate = successes / n_samples
    avg_error = np.mean(errors)
    
    return success_rate, avg_error

def analyze_patterns_by_sparsity():
    """Analyze factorization patterns for each sparsity level."""
    print("ANALYZING FACTORIZATION PATTERNS BY SPARSITY")
    print("="*60)
    
    results = {}
    
    # Analyze each sparsity level
    for n_zeros in range(13):  # 0 to 12 zeros
        print(f"\nAnalyzing patterns with {n_zeros} zeros...")
        
        patterns = generate_all_patterns(n_zeros)
        n_patterns = len(patterns)
        
        # Sample patterns if too many
        if n_patterns > 100:
            print(f"  Sampling 100 out of {n_patterns} patterns")
            sampled_patterns = np.random.choice(patterns, size=100, replace=False)
        else:
            sampled_patterns = patterns
        
        pattern_results = []
        
        for pattern in tqdm(sampled_patterns, desc=f"  Testing {n_zeros} zeros"):
            success_rate, avg_error = test_pattern_factorizability(pattern, n_samples=5)
            pattern_results.append({
                'pattern': pattern,
                'success_rate': success_rate,
                'avg_error': avg_error,
                'matrix_viz': pattern_to_matrix_viz(pattern)
            })
        
        # Sort by success rate
        pattern_results.sort(key=lambda x: x['success_rate'], reverse=True)
        results[n_zeros] = pattern_results
        
        # Summary statistics
        success_rates = [r['success_rate'] for r in pattern_results]
        print(f"  Success rates: min={min(success_rates):.0%}, "
              f"max={max(success_rates):.0%}, "
              f"mean={np.mean(success_rates):.0%}")
    
    return results

def visualize_pattern_gallery(results):
    """Create a gallery showing factorizable vs non-factorizable patterns."""
    fig, axes = plt.subplots(5, 3, figsize=(18, 24))
    fig.suptitle('Factorization Patterns by Sparsity Level', fontsize=16)
    
    # Color maps
    cmap_success = ListedColormap(['white', 'darkgreen'])
    cmap_fail = ListedColormap(['white', 'darkred'])
    
    for idx, n_zeros in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col] if idx < 12 else axes[4, 0]
        
        if n_zeros in results:
            pattern_data = results[n_zeros]
            
            # Find best and worst patterns
            best_patterns = [p for p in pattern_data if p['success_rate'] > 0.5][:3]
            worst_patterns = [p for p in pattern_data if p['success_rate'] < 0.5][-3:]
            
            # Create visualization
            ax.text(0.5, 0.95, f'{n_zeros} zeros', 
                   ha='center', va='top', transform=ax.transAxes, 
                   fontsize=14, fontweight='bold')
            
            # Show statistics
            all_rates = [p['success_rate'] for p in pattern_data]
            ax.text(0.5, 0.88, f'Overall: {np.mean(all_rates):.0%} factorizable', 
                   ha='center', va='top', transform=ax.transAxes, fontsize=10)
            
            # Plot best patterns
            y_pos = 0.75
            if best_patterns:
                ax.text(0.1, y_pos, 'Best patterns:', fontsize=10, transform=ax.transAxes)
                y_pos -= 0.15
                
                for i, p in enumerate(best_patterns[:2]):
                    # Create small matrix visualization
                    mat_ax = ax.inset_axes([0.1 + i*0.25, y_pos-0.15, 0.15, 0.15])
                    mat_ax.imshow(p['matrix_viz'], cmap=cmap_success, vmin=0, vmax=1)
                    mat_ax.set_xticks([])
                    mat_ax.set_yticks([])
                    mat_ax.text(0.5, -0.2, f"{p['success_rate']:.0%}", 
                               ha='center', transform=mat_ax.transAxes, fontsize=8)
            
            # Plot worst patterns
            y_pos = 0.35
            if worst_patterns and n_zeros not in [0, 12]:  # Skip if all are good
                ax.text(0.1, y_pos, 'Worst patterns:', fontsize=10, transform=ax.transAxes)
                y_pos -= 0.15
                
                for i, p in enumerate(worst_patterns[:2]):
                    mat_ax = ax.inset_axes([0.1 + i*0.25, y_pos-0.15, 0.15, 0.15])
                    mat_ax.imshow(p['matrix_viz'], cmap=cmap_fail, vmin=0, vmax=1)
                    mat_ax.set_xticks([])
                    mat_ax.set_yticks([])
                    mat_ax.text(0.5, -0.2, f"{p['success_rate']:.0%}", 
                               ha='center', transform=mat_ax.transAxes, fontsize=8)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
    
    # Clear unused axes
    if idx < 14:
        for i in range(idx + 1, 15):
            row = i // 3
            col = i % 3
            if row < 5 and col < 3:
                axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('factorization_pattern_gallery.png', dpi=300, bbox_inches='tight')
    print("\nSaved pattern gallery to factorization_pattern_gallery.png")

def analyze_pattern_structure(results):
    """Analyze what structural features predict factorizability."""
    print("\n" + "="*60)
    print("STRUCTURAL ANALYSIS")
    print("="*60)
    
    # Collect all patterns with their features
    all_patterns = []
    
    for n_zeros, patterns in results.items():
        for p in patterns:
            pattern = p['pattern']
            matrix = p['matrix_viz']
            
            # Extract features
            features = {
                'n_zeros': n_zeros,
                'success_rate': p['success_rate'],
                'pattern': pattern,
                
                # Diagonal zeros
                'diag_zeros': sum(1 for i in range(4) if pattern & (1 << (5*i))),
                
                # Row/column structure
                'empty_rows': sum(1 for i in range(4) if all(matrix[i, j] == 0 for j in range(4))),
                'empty_cols': sum(1 for j in range(4) if all(matrix[i, j] == 0 for i in range(4))),
                
                # Block structure
                'upper_left_zeros': sum(1 for i in range(2) for j in range(2) if matrix[i, j] == 0),
                'upper_right_zeros': sum(1 for i in range(2) for j in range(2, 4) if matrix[i, j] == 0),
                'lower_left_zeros': sum(1 for i in range(2, 4) for j in range(2) if matrix[i, j] == 0),
                'lower_right_zeros': sum(1 for i in range(2, 4) for j in range(2, 4) if matrix[i, j] == 0),
                
                # Symmetry
                'symmetric': np.allclose(matrix, matrix.T),
                'antisymmetric': np.allclose(matrix, -matrix.T),
            }
            
            all_patterns.append(features)
    
    # Analyze correlations with factorizability
    print("\nFeatures correlated with factorizability:")
    
    # Group by feature and compute average success rate
    feature_names = ['diag_zeros', 'empty_rows', 'empty_cols', 
                     'upper_left_zeros', 'upper_right_zeros', 
                     'lower_left_zeros', 'lower_right_zeros',
                     'symmetric', 'antisymmetric']
    
    for feature in feature_names:
        feature_groups = defaultdict(list)
        for p in all_patterns:
            feature_groups[p[feature]].append(p['success_rate'])
        
        print(f"\n{feature}:")
        for value in sorted(feature_groups.keys()):
            rates = feature_groups[value]
            if len(rates) > 5:  # Only show if enough samples
                print(f"  {value}: {np.mean(rates):.0%} factorizable (n={len(rates)})")
    
    return all_patterns

def create_heatmap_visualization(results):
    """Create heatmap showing factorizability by pattern type."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Create matrix showing success rates
    # Rows: number of zeros
    # Columns: pattern characteristics
    
    # Define pattern types we care about
    pattern_types = [
        ('All zeros diagonal', lambda p: p['diag_zeros'] == min(4, p['n_zeros'])),
        ('Zeros in one row', lambda p: p['empty_rows'] >= 1),
        ('Zeros in one column', lambda p: p['empty_cols'] >= 1),
        ('Block diagonal zeros', lambda p: p['upper_right_zeros'] + p['lower_left_zeros'] == p['n_zeros']),
        ('Symmetric pattern', lambda p: p['symmetric']),
        ('Random pattern', lambda p: not any([
            p['diag_zeros'] == min(4, p['n_zeros']),
            p['empty_rows'] >= 1,
            p['empty_cols'] >= 1,
            p['symmetric']
        ]))
    ]
    
    # Build heatmap data
    heatmap_data = np.zeros((13, len(pattern_types)))
    counts = np.zeros((13, len(pattern_types)))
    
    all_patterns = []
    for n_zeros, patterns in results.items():
        for p in patterns:
            features = {
                'n_zeros': n_zeros,
                'success_rate': p['success_rate'],
                'pattern': p['pattern'],
                'matrix_viz': p['matrix_viz'],
                'diag_zeros': sum(1 for i in range(4) if p['pattern'] & (1 << (5*i))),
                'empty_rows': sum(1 for i in range(4) if all(p['matrix_viz'][i, j] == 0 for j in range(4))),
                'empty_cols': sum(1 for j in range(4) if all(p['matrix_viz'][i, j] == 0 for i in range(4))),
                'upper_right_zeros': sum(1 for i in range(2) for j in range(2, 4) if p['matrix_viz'][i, j] == 0),
                'lower_left_zeros': sum(1 for i in range(2, 4) for j in range(2) if p['matrix_viz'][i, j] == 0),
                'symmetric': np.allclose(p['matrix_viz'], p['matrix_viz'].T),
            }
            
            for j, (name, condition) in enumerate(pattern_types):
                if condition(features):
                    heatmap_data[n_zeros, j] += features['success_rate']
                    counts[n_zeros, j] += 1
    
    # Average the success rates
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_data = np.divide(heatmap_data, counts)
        heatmap_data[np.isnan(heatmap_data)] = -1  # Mark as no data
    
    # Create heatmap
    masked_data = np.ma.masked_where(heatmap_data < 0, heatmap_data)
    
    im = ax.imshow(masked_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(pattern_types)))
    ax.set_xticklabels([name for name, _ in pattern_types], rotation=45, ha='right')
    ax.set_yticks(range(13))
    ax.set_yticklabels([f'{i} zeros' for i in range(13)])
    
    # Add text annotations
    for i in range(13):
        for j in range(len(pattern_types)):
            if counts[i, j] > 0:
                text = f'{heatmap_data[i, j]:.0%}\n(n={int(counts[i, j])})'
                ax.text(j, i, text, ha='center', va='center', fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Factorization Success Rate', rotation=270, labelpad=20)
    
    ax.set_title('Factorizability by Zero Pattern Type', fontsize=14)
    ax.set_xlabel('Pattern Type', fontsize=12)
    ax.set_ylabel('Number of Zeros', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('factorization_heatmap_patterns.png', dpi=300, bbox_inches='tight')
    print("\nSaved heatmap to factorization_heatmap_patterns.png")

def find_special_patterns(results):
    """Find particularly interesting patterns."""
    print("\n" + "="*60)
    print("SPECIAL PATTERNS")
    print("="*60)
    
    # Find surprising patterns
    for n_zeros in [2, 3, 4, 5, 6, 7, 8]:  # The "death valley"
        if n_zeros not in results:
            continue
            
        patterns = results[n_zeros]
        
        # Find any that ARE factorizable
        good_patterns = [p for p in patterns if p['success_rate'] > 0.5]
        
        if good_patterns:
            print(f"\nSurprisingly factorizable patterns with {n_zeros} zeros:")
            for p in good_patterns[:3]:
                print(f"  Pattern {bin(p['pattern'])[2:].zfill(16)}: {p['success_rate']:.0%} success")
                print("  Matrix visualization:")
                for row in p['matrix_viz']:
                    print("    " + " ".join("█" if x == 1 else "·" for x in row))

if __name__ == "__main__":
    # Analyze patterns
    results = analyze_patterns_by_sparsity()
    
    # Create visualizations
    visualize_pattern_gallery(results)
    
    # Structural analysis
    all_patterns = analyze_pattern_structure(results)
    
    # Create heatmap
    create_heatmap_visualization(results)
    
    # Find special cases
    find_special_patterns(results)
    
    print("\n✓ Analysis complete!")