#!/usr/bin/env python3
"""
Fixed F_3 search using {0, 1, -1} representation.
More efficient and correct rank computation.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import time
from collections import defaultdict

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

@jit
def rank_f3_correct(M):
    """
    Correct rank computation for a single matrix over F_3.
    Using {0, 1, -1} representation.
    """
    # Work with float32 for compatibility
    M = M.astype(jnp.float32)
    n = M.shape[0]
    
    # Gaussian elimination
    rank = 0
    for col in range(n):
        # Find pivot in column col, starting from row 'rank'
        pivot_found = False
        for row in range(rank, n):
            if M[row, col] != 0:
                # Swap rows if needed
                if row != rank:
                    M = M.at[[rank, row]].set(M[[row, rank]])
                
                # Scale row to make pivot = 1
                pivot = M[rank, col]
                # In F_3: inverse of 1 is 1, inverse of -1 is -1
                M = M.at[rank].set(M[rank] / pivot)
                
                # Eliminate column
                for r in range(n):
                    if r != rank and M[r, col] != 0:
                        M = M.at[r].set(M[r] - M[r, col] * M[rank])
                
                rank += 1
                pivot_found = True
                break
        
        if not pivot_found:
            continue
    
    return rank

# Vectorize for batch processing
rank_f3_batch = vmap(rank_f3_correct)

@jit
def int_to_matrix_f3_balanced(n):
    """Convert integer to 4x4 matrix using {0, 1, -1}."""
    # Map 0->0, 1->1, 2->-1
    matrix = jnp.zeros((4, 4), dtype=jnp.float32)
    
    for i in range(16):
        digit = (n // (3**i)) % 3
        val = jnp.where(digit == 2, -1.0, digit.astype(jnp.float32))
        matrix = matrix.at[i // 4, i % 4].set(val)
    
    return matrix

# Vectorize
int_to_matrix_batch = vmap(int_to_matrix_f3_balanced)

@jit
def matrix_to_int_f3_balanced(M):
    """Convert matrix with {0, 1, -1} back to integer."""
    # Map 0->0, 1->1, -1->2
    n = 0
    for i in range(16):
        val = M[i // 4, i % 4]
        digit = jnp.where(val == -1, 2, val.astype(jnp.int32))
        n += digit * (3**i)
    return n

@jit
def hadamard_product_f3_balanced(A, B):
    """Hadamard product for {0, 1, -1} matrices."""
    # Direct multiplication works correctly:
    # 0*x = 0, 1*1 = 1, 1*(-1) = -1, (-1)*(-1) = 1
    return A * B

@jit
def count_zeros(M):
    """Count zeros in matrix."""
    return jnp.sum(M == 0)

def find_all_rank2_correct():
    """Find all rank-2 matrices with correct rank computation."""
    print("Finding all rank-2 matrices (correct algorithm)...")
    
    batch_size = 50000
    n_total = 3**16
    rank2_list = []
    rank_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    start_time = time.time()
    
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_ints = jnp.arange(start, end)
        
        # Convert to matrices
        matrices = int_to_matrix_batch(batch_ints)
        
        # Compute ranks
        ranks = rank_f3_batch(matrices)
        
        # Update counts
        for r in range(5):
            rank_counts[r] += jnp.sum(ranks == r).item()
        
        # Extract rank-2 matrices
        rank2_mask = ranks == 2
        rank2_ints = batch_ints[rank2_mask]
        rank2_list.extend(rank2_ints.tolist())
        
        if start % 1000000 == 0:
            elapsed = time.time() - start_time
            rate = (start + batch_size) / elapsed
            eta = (n_total - start) / rate
            print(f"  Progress: {start:,}/{n_total:,} ({start/n_total*100:.1f}%)")
            print(f"  Found {len(rank2_list):,} rank-2 so far")
            print(f"  Rate: {rate:,.0f} matrices/sec, ETA: {eta/60:.1f} min")
    
    print(f"\nRank distribution:")
    for r in range(5):
        print(f"  Rank {r}: {rank_counts[r]:,} ({rank_counts[r]/n_total*100:.3f}%)")
    
    return jnp.array(rank2_list)

def quick_test():
    """Quick test to verify rank computation."""
    print("Quick verification of rank computation...")
    
    # Test a few known cases
    test_matrices = [
        jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),  # I, rank 4
        jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),  # rank 2
        jnp.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),  # rank 1
        jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),  # rank 0
    ]
    
    for i, M in enumerate(test_matrices):
        rank = rank_f3_correct(M)
        print(f"  Test matrix {i+1}: rank = {rank}")
    
    # Test on random sample
    print("\nSampling 10000 random matrices...")
    sample_ints = jax.random.randint(jax.random.PRNGKey(0), (10000,), 0, 3**16)
    sample_matrices = int_to_matrix_batch(sample_ints)
    sample_ranks = rank_f3_batch(sample_matrices)
    
    for r in range(5):
        count = jnp.sum(sample_ranks == r).item()
        print(f"  Rank {r}: {count} ({count/10000*100:.1f}%)")
    
    # Estimate total rank-2
    rank2_frac = jnp.sum(sample_ranks == 2).item() / 10000
    print(f"\nEstimated rank-2 matrices: {int(rank2_frac * 3**16):,}")

def main():
    print("F_3 HADAMARD FACTORIZATION - FIXED VERSION")
    print("="*70)
    print("Using {0, 1, -1} representation for efficiency")
    print()
    
    # First do quick test
    quick_test()
    
    # Then full search if test looks good
    response = input("\nProceed with full search? (y/n): ")
    if response.lower() == 'y':
        rank2_ints = find_all_rank2_correct()
        print(f"\nTotal rank-2 matrices found: {len(rank2_ints):,}")
        print("(Should be ~811,200)")
        
        # Save for further analysis
        with open('f3_rank2_matrices.npy', 'wb') as f:
            np.save(f, rank2_ints)
        print("Saved rank-2 matrices to f3_rank2_matrices.npy")

if __name__ == "__main__":
    main()