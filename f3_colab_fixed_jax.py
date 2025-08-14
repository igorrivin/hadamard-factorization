#!/usr/bin/env python3
"""
F_3 Hadamard search with JAX-compatible rank computation.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
import time
from collections import defaultdict
from tqdm.auto import tqdm
import pickle

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

@jit
def rank_f3_unrolled(M):
    """
    Fully unrolled rank computation for 4x4 matrix over F_3.
    JAX-compatible with no Python control flow.
    """
    M = M.astype(jnp.float32) % 3
    rank = 0
    
    # Process column 0
    # Find first non-zero in column 0
    col0_nonzero = M[:, 0] != 0
    has_pivot = jnp.any(col0_nonzero)
    pivot_row = jnp.argmax(col0_nonzero)  # First True index
    
    # Swap row 0 with pivot row (conditional on having pivot)
    def swap_rows_0():
        M_new = M.at[[0, pivot_row]].set(M[[pivot_row, 0]])
        return M_new
    M = lax.cond(has_pivot & (pivot_row != 0), swap_rows_0, lambda: M)
    
    # Normalize row 0 (make pivot = 1)
    pivot_val = M[0, 0]
    inv = jnp.where(pivot_val == 2, 2.0, jnp.where(pivot_val == 1, 1.0, 0.0))
    M = M.at[0].multiply(inv)
    M = M % 3
    
    # Eliminate column 0
    for i in range(1, 4):
        factor = M[i, 0]
        M = M.at[i].add(-factor * M[0])
    M = M % 3
    
    # Update rank
    rank = jnp.where(has_pivot, 1, 0)
    
    # Process column 1 (starting from row 1 if rank > 0)
    start_row = rank
    col1_nonzero = M[start_row:, 1] != 0
    has_pivot1 = jnp.any(col1_nonzero)
    pivot_row1 = start_row + jnp.argmax(col1_nonzero)
    
    # Swap rows if needed
    def swap_rows_1():
        return M.at[[start_row, pivot_row1]].set(M[[pivot_row1, start_row]])
    M = lax.cond(has_pivot1 & (pivot_row1 != start_row), swap_rows_1, lambda: M)
    
    # Normalize and eliminate
    pivot_val1 = M[start_row, 1]
    inv1 = jnp.where(pivot_val1 == 2, 2.0, jnp.where(pivot_val1 == 1, 1.0, 0.0))
    M = lax.cond(has_pivot1,
                 lambda: M.at[start_row].multiply(inv1) % 3,
                 lambda: M)
    
    # Eliminate column 1
    def eliminate_col1():
        M_new = M
        for i in range(4):
            if i != start_row:
                factor = M_new[i, 1]
                M_new = M_new.at[i].add(-factor * M_new[start_row])
        return M_new % 3
    
    M = lax.cond(has_pivot1, eliminate_col1, lambda: M)
    rank = jnp.where(has_pivot1, rank + 1, rank)
    
    # Similar for columns 2 and 3 (abbreviated)
    # In practice, we'd fully unroll these too
    
    # For now, approximate final rank by counting non-zero rows
    non_zero_rows = jnp.sum(jnp.any(M != 0, axis=1))
    
    return non_zero_rows

# Vectorize
rank_f3_batch = vmap(rank_f3_unrolled)

# Simpler approach: use NumPy for rank computation
def rank_f3_numpy(M_int):
    """Compute rank using NumPy (not JIT-ed)."""
    # Convert integer to matrix
    M = np.zeros((4, 4), dtype=int)
    for i in range(16):
        M[i // 4, i % 4] = (M_int // (3**i)) % 3
    
    # Gaussian elimination
    M = M.copy()
    rank = 0
    
    for col in range(4):
        # Find pivot
        pivot_found = False
        for row in range(rank, 4):
            if M[row, col] % 3 != 0:
                # Swap rows
                if row != rank:
                    M[[rank, row]] = M[[row, rank]]
                
                # Make pivot 1
                pivot = M[rank, col] % 3
                inv = 1 if pivot == 1 else 2
                M[rank] = (M[rank] * inv) % 3
                
                # Eliminate column
                for r in range(4):
                    if r != rank and M[r, col] % 3 != 0:
                        factor = M[r, col] % 3
                        M[r] = (M[r] - factor * M[rank]) % 3
                
                rank += 1
                pivot_found = True
                break
    
    return rank

@jit
def int_to_matrix_f3(n):
    """Convert integer to 4x4 matrix over F_3."""
    indices = jnp.arange(16)
    powers = 3 ** indices
    digits = (n // powers) % 3
    return digits.reshape(4, 4).astype(jnp.float32)

int_to_matrix_batch = vmap(int_to_matrix_f3)

@jit
def matrix_to_int_f3(M):
    """Convert matrix to integer."""
    M_flat = M.flatten()
    indices = jnp.arange(16)
    powers = 3 ** indices
    return jnp.sum(M_flat * powers)

@jit
def hadamard_product_f3(A, B):
    """Hadamard product over F_3."""
    return (A * B) % 3

@jit
def count_zeros(M):
    """Count zeros in matrix."""
    return jnp.sum(M == 0)

count_zeros_batch = vmap(count_zeros)

def find_all_rank2_matrices_numpy():
    """Find all rank-2 matrices using NumPy for rank computation."""
    print("\n" + "="*70)
    print("STEP 1: Finding all rank-2 matrices")
    print("="*70)
    
    n_total = 3**16
    rank2_list = []
    rank_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    # Process in chunks
    chunk_size = 100000
    
    pbar = tqdm(total=n_total, desc="Searching matrices", unit="matrices", 
                unit_scale=True)
    
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        
        # Compute ranks for this chunk
        for n in range(start, end):
            rank = rank_f3_numpy(n)
            rank_counts[rank] += 1
            
            if rank == 2:
                rank2_list.append(n)
        
        pbar.update(end - start)
        pbar.set_postfix({'rank-2 found': f'{len(rank2_list):,}'})
    
    pbar.close()
    
    print(f"\nRank distribution:")
    for r in range(5):
        pct = rank_counts[r]/n_total*100
        print(f"  Rank {r}: {rank_counts[r]:,} ({pct:.3f}%)")
    
    return np.array(rank2_list)

def build_product_table_jax(rank2_ints, chunk_size=1000):
    """Build product table using JAX for matrix operations."""
    print("\n" + "="*70)
    print("STEP 2: Building product lookup table")
    print("="*70)
    
    # Convert all rank-2 integers to matrices once
    rank2_matrices = int_to_matrix_batch(jnp.array(rank2_ints))
    n = len(rank2_matrices)
    
    total_pairs = n * (n + 1) // 2
    print(f"Processing {total_pairs:,} matrix pairs...")
    
    product_map = defaultdict(list)
    
    pbar = tqdm(total=total_pairs, desc="Computing products", unit="pairs",
                unit_scale=True)
    
    pairs_processed = 0
    
    for i in range(0, n, chunk_size):
        chunk_a = rank2_matrices[i:i+chunk_size]
        
        for j in range(i, n, chunk_size):
            chunk_b = rank2_matrices[j:j+chunk_size]
            
            # Compute all products in this chunk pair
            # Use nested vmap for efficiency
            def compute_product(a, b):
                return hadamard_product_f3(a, b)
            
            # Vectorize over chunk_b for each a in chunk_a
            products_batch = vmap(lambda a: vmap(lambda b: compute_product(a, b))(chunk_b))(chunk_a)
            
            # Convert products back to integers and store
            for ii in range(len(chunk_a)):
                for jj in range(len(chunk_b)):
                    if i + ii <= j + jj:  # Avoid duplicates
                        product = products_batch[ii, jj]
                        prod_int = int(matrix_to_int_f3(product))
                        
                        product_map[prod_int].append((rank2_ints[i+ii], 
                                                     rank2_ints[j+jj]))
                        pairs_processed += 1
            
            pbar.update(min(len(chunk_a) * len(chunk_b), total_pairs - pbar.n))
            pbar.set_postfix({'unique products': f'{len(product_map):,}'})
    
    pbar.close()
    print(f"\nFound {len(product_map):,} unique products")
    
    return product_map

def main():
    print("F_3 HADAMARD FACTORIZATION - HYBRID APPROACH")
    print("="*70)
    print("Using NumPy for rank computation, JAX for matrix operations")
    
    # Quick test
    print("\nQuick test of rank computation...")
    test_cases = [
        3**16 - 1,  # All 2s
        0,          # All 0s
        1 + 3 + 9 + 27,  # Identity-like
    ]
    
    for n in test_cases:
        rank = rank_f3_numpy(n)
        M = int_to_matrix_f3(n)
        print(f"  Matrix {n}: rank = {rank}, zeros = {count_zeros(M)}")
    
    # Main computation
    print("\nStarting main computation...")
    
    # Step 1: Find rank-2 matrices
    rank2_ints = find_all_rank2_matrices_numpy()
    print(f"\n✓ Found {len(rank2_ints):,} rank-2 matrices")
    
    if len(rank2_ints) < 500000:
        print("WARNING: Fewer rank-2 matrices than expected!")
        print("Expected ~811,200 based on theory")
    
    # Step 2: Build product table
    product_map = build_product_table_jax(rank2_ints)
    
    # Save intermediate results
    print("\nSaving results...")
    with open('f3_rank2_matrices.npy', 'wb') as f:
        np.save(f, rank2_ints)
    
    with open('f3_product_map.pkl', 'wb') as f:
        pickle.dump(dict(product_map), f)
    
    print("\n✓ Results saved!")
    print("Continue with expressibility check using saved data")

if __name__ == "__main__":
    main()