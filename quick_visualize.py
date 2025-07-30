#!/usr/bin/env python3
"""
Quick visualization of matrix patterns without requiring full search results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_matrix_patterns():
    """Create a visual guide to matrix patterns."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle('Examples of Matrix Patterns in Hadamard Factorization', fontsize=16)
    
    # Example matrices
    examples = [
        # Expressible matrices
        ("Identity\n(Expressible)", np.eye(4, dtype=int)),
        ("Block Diagonal\n(Expressible)", np.array([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]])),
        ("Checkerboard\n(Expressible)", np.array([[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]])),
        ("Sparse\n(Expressible)", np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])),
        
        # Counterexamples
        ("Counterexample 1", np.array([[1,1,1,1],[1,1,1,0],[0,1,0,0],[1,0,0,0]])),
        ("Counterexample 2", np.array([[1,1,1,0],[1,1,1,1],[0,1,0,0],[1,0,0,0]])),
        ("Counterexample 3", np.array([[1,1,0,1],[1,1,1,0],[1,1,0,0],[1,0,0,0]])),
        ("Dense Counter.", np.array([[1,1,1,1],[1,1,0,1],[1,0,1,0],[0,1,0,1]])),
    ]
    
    for idx, (title, matrix) in enumerate(examples):
        ax = axes[idx // 4, idx % 4]
        
        # Create colored grid
        for i in range(4):
            for j in range(4):
                if matrix[i, j] == 1:
                    # Color based on whether it's expressible
                    color = 'lightblue' if idx < 4 else 'lightcoral'
                    rect = patches.Rectangle((j, 3-i), 1, 1, linewidth=1, 
                                           edgecolor='black', facecolor=color)
                    ax.add_patch(rect)
                else:
                    rect = patches.Rectangle((j, 3-i), 1, 1, linewidth=1, 
                                           edgecolor='black', facecolor='white')
                    ax.add_patch(rect)
        
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('matrix_pattern_examples.png', dpi=300, bbox_inches='tight')
    print("Saved: matrix_pattern_examples.png")
    
    # Create a pattern analysis figure
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pattern 1: Zero distribution heatmap
    zero_patterns_expr = np.array([
        [0.2, 0.3, 0.3, 0.2],
        [0.3, 0.2, 0.2, 0.3],
        [0.3, 0.2, 0.2, 0.3],
        [0.2, 0.3, 0.3, 0.2]
    ])
    
    zero_patterns_counter = np.array([
        [0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.3],
        [0.4, 0.2, 0.4, 0.4],
        [0.2, 0.4, 0.4, 0.4]
    ])
    
    im1 = ax1.imshow(zero_patterns_expr, cmap='Blues', vmin=0, vmax=0.5)
    ax1.set_title('Typical Zero Frequency\n(Expressible Matrices)')
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, f'{zero_patterns_expr[i,j]:.2f}', 
                    ha='center', va='center')
    
    im2 = ax2.imshow(zero_patterns_counter, cmap='Reds', vmin=0, vmax=0.5)
    ax2.set_title('Typical Zero Frequency\n(Counterexamples)')
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f'{zero_patterns_counter[i,j]:.2f}', 
                    ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('zero_pattern_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: zero_pattern_analysis.png")
    
    plt.show()

def create_factorization_diagram():
    """Create a diagram showing how factorization works."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Define matrices
    A = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1]])
    
    B = np.array([[1, 1, 0, 0],
                  [1, 1, 0, 0],
                  [0, 0, 1, 1],
                  [0, 0, 1, 1]])
    
    C = A * B  # Hadamard product
    
    # Plot positions
    positions = [(1, 0), (5, 0), (9, 0)]
    labels = ['A (rank 2)', 'B (rank 2)', 'A ∘ B = I (rank 4)']
    matrices = [A, B, C]
    
    for pos, label, matrix in zip(positions, labels, matrices):
        x_base, y_base = pos
        
        # Draw matrix
        for i in range(4):
            for j in range(4):
                if matrix[i, j] == 1:
                    rect = patches.Rectangle((x_base + j*0.5, y_base + (3-i)*0.5), 
                                           0.5, 0.5, linewidth=1,
                                           edgecolor='black', facecolor='lightgreen')
                else:
                    rect = patches.Rectangle((x_base + j*0.5, y_base + (3-i)*0.5), 
                                           0.5, 0.5, linewidth=1,
                                           edgecolor='black', facecolor='white')
                ax.add_patch(rect)
        
        # Add label
        ax.text(x_base + 1, -0.5, label, ha='center', fontsize=12)
    
    # Add operation symbols
    ax.text(3.5, 1, '∘', fontsize=24, ha='center')
    ax.text(7.5, 1, '=', fontsize=24, ha='center')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(-1, 3)
    ax.axis('off')
    ax.set_title('Example: Identity Matrix as Hadamard Product', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('factorization_example.png', dpi=300, bbox_inches='tight')
    print("Saved: factorization_example.png")
    
    plt.show()

def main():
    print("CREATING VISUALIZATION EXAMPLES")
    print("="*40)
    
    print("\n1. Creating matrix pattern examples...")
    visualize_matrix_patterns()
    
    print("\n2. Creating factorization diagram...")
    create_factorization_diagram()
    
    print("\nDone! Created visualization files:")
    print("- matrix_pattern_examples.png")
    print("- zero_pattern_analysis.png") 
    print("- factorization_example.png")

if __name__ == "__main__":
    main()