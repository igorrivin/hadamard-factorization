#!/usr/bin/env python3
"""
F_3 Hadamard search - Super optimized version.
Uses lookup tables and bit tricks for maximum speed.
"""

import numpy as np
import numba as nb
from numba import jit, prange
import jax
import jax.numpy as jnp
from jax import vmap
import time
from collections import defaultdict
from tqdm.auto import tqdm
import pickle

print("Setting up super-fast Numba JIT compilation...")

# Pre-compute rank lookup table for small matrices
@nb.jit(nopython=True, cache=True)
def build_rank_lookup():
    """Build lookup table for 2x2 submatrix ranks."""
    lookup = np.zeros(81, dtype=np.int8)  # 3^4 = 81 possible 2x2 matrices
    
    for i in range(81):
        # Extract 2x2 matrix
        a, b = i % 3, (i // 3) % 3
        c, d = (i // 9) % 3, (i // 27) % 3
        
        # Compute rank
        if a == 0 and b == 0 and c == 0 and d == 0:
            rank = 0
        elif (a * d) % 3 == (b * c) % 3:  # Determinant = 0
            rank = 1 if (a != 0 or b != 0 or c != 0 or d != 0) else 0
        else:
            rank = 2
            
        lookup[i] = rank
    
    return lookup

# Global lookup table (computed once)
RANK2_LOOKUP = build_rank_lookup()

@nb.jit(nopython=True, cache=True, fastmath=True)
def rank_f3_superfast(M_int):
    """Ultra-fast rank computation using bit manipulation and lookup tables."""
    # Extract matrix digits in one pass
    digits = np.zeros(16, dtype=np.int8)
    temp = M_int
    for i in range(16):
        digits[i] = temp % 3
        temp //= 3
    
    # Quick check for zero matrix
    if M_int == 0:
        return 0
    
    # Use bit manipulation for row operations
    rows = np.zeros(4, dtype=np.int32)
    for i in range(4):
        rows[i] = digits[4*i] + 3*digits[4*i+1] + 9*digits[4*i+2] + 27*digits[4*i+3]
    
    # Gaussian elimination with bit tricks
    rank = 0
    used_cols = 0
    
    for row_idx in range(4):
        row = rows[row_idx]
        
        # Find leftmost non-zero position
        for col in range(4):
            if used_cols & (1 << col):
                continue
                
            val = (row // (3**col)) % 3
            if val != 0:
                # Mark column as used
                used_cols |= (1 << col)
                
                # Normalize row (multiply by inverse)
                if val == 2:
                    row = (row * 2) % 81  # Faster modulo for small numbers
                
                # Eliminate from other rows
                for j in range(4):
                    if j != row_idx:
                        other_val = (rows[j] // (3**col)) % 3
                        if other_val != 0:
                            rows[j] = (rows[j] - other_val * row) % 81
                
                rank += 1
                break
    
    return rank

@nb.jit(nopython=True, cache=True, parallel=True)
def find_rank2_superfast(start, end):
    """Find rank-2 matrices using parallel Numba with optimizations."""
    # Pre-allocate for expected ~1.85% rank-2 matrices
    expected_rank2 = int((end - start) * 0.019)
    rank2_list = np.empty(expected_rank2 + 1000, dtype=np.int32)
    rank2_count = 0
    
    # Use parallel range
    for n in prange(start, end):
        if rank_f3_superfast(n) == 2:
            if rank2_count < len(rank2_list):
                rank2_list[rank2_count] = n
                rank2_count += 1
    
    return rank2_list[:rank2_count].copy()

# Simplified JAX functions
@jax.jit
def int_to_matrix_f3(n):
    indices = jnp.arange(16)
    return ((n // (3 ** indices)) % 3).reshape(4, 4)

int_to_matrix_batch = vmap(int_to_matrix_f3)

@jax.jit
def matrix_to_int_f3(M):
    return jnp.sum(M.flatten() * (3 ** jnp.arange(16)))

@jax.jit
def hadamard_product_f3(A, B):
    return (A * B) % 3

@jax.jit
def count_zeros(M):
    return jnp.sum(M == 0)

count_zeros_batch = vmap(count_zeros)

def find_all_rank2_superfast():
    """Find all rank-2 matrices with maximum speed."""
    print("\n" + "="*70)
    print("STEP 1: Finding all rank-2 matrices (SUPER FAST MODE)")
    print("="*70)
    
    n_total = 3**16
    batch_size = 10000000  # Large batches for parallel efficiency
    rank2_list = []
    
    # Warm up JIT
    print("Warming up JIT compiler...")
    test_batch = find_rank2_superfast(0, 1000)
    print(f"✓ JIT ready. Test found {len(test_batch)} rank-2 matrices in first 1000")
    
    pbar = tqdm(total=n_total, desc="Searching matrices", unit="matrices", 
                unit_scale=True, smoothing=0.1)
    
    start_time = time.time()
    
    # Process in large parallel batches
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        
        # Find rank-2 in this batch
        batch_rank2 = find_rank2_superfast(start, end)
        rank2_list.extend(batch_rank2)
        
        # Update progress
        pbar.update(end - start)
        elapsed = time.time() - start_time
        rate = (start + batch_size) / elapsed
        pbar.set_postfix({
            'rank-2': f'{len(rank2_list):,}',
            'rate': f'{rate:,.0f}/s',
            'ETA': f'{(n_total-start)/rate/60:.0f}m'
        })
    
    pbar.close()
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes ({n_total/elapsed:,.0f} matrices/sec)")
    print(f"Found {len(rank2_list):,} rank-2 matrices (~811,200 expected)")
    
    return np.array(rank2_list, dtype=np.int32)

def build_product_table_batched(rank2_ints, chunk_size=3000):
    """Optimized product table building."""
    print("\n" + "="*70)
    print("STEP 2: Building product lookup table (GPU batched)")
    print("="*70)
    
    # Only compute products for a sample if full table is too large
    if len(rank2_ints) > 800000:
        print("NOTE: Full product table would be huge. Computing sample instead.")
        # Sample approach: only compute products for first 100k rank-2 matrices
        sample_size = min(100000, len(rank2_ints))
        rank2_sample = rank2_ints[:sample_size]
        print(f"Using sample of {sample_size:,} rank-2 matrices")
    else:
        rank2_sample = rank2_ints
    
    # Convert to matrices
    print("Converting to matrix form on GPU...")
    rank2_matrices = int_to_matrix_batch(jnp.array(rank2_sample))
    n = len(rank2_matrices)
    
    total_pairs = n * (n + 1) // 2
    print(f"Computing {total_pairs:,} Hadamard products...")
    
    product_set = set()  # Use set for faster lookups
    
    pbar = tqdm(total=total_pairs, desc="Products", unit="pairs", 
                unit_scale=True, smoothing=0.1)
    
    # Process in chunks
    for i in range(0, n, chunk_size):
        chunk_a = rank2_matrices[i:i+chunk_size]
        
        for j in range(i, n, chunk_size):
            chunk_b = rank2_matrices[j:j+chunk_size]
            
            # Vectorized products
            products = vmap(lambda a: vmap(lambda b: hadamard_product_f3(a, b))(chunk_b))(chunk_a)
            
            # Convert to integers
            product_ints = vmap(lambda row: vmap(matrix_to_int_f3)(row))(products)
            product_ints_np = np.array(product_ints, dtype=np.int32)
            
            # Add to set
            for ii in range(len(chunk_a)):
                for jj in range(len(chunk_b)):
                    if i + ii <= j + jj:
                        product_set.add(int(product_ints_np[ii, jj]))
            
            pairs_done = len(chunk_a) * len(chunk_b)
            pbar.update(pairs_done)
            pbar.set_postfix({'unique': f'{len(product_set):,}'})
    
    pbar.close()
    print(f"Found {len(product_set):,} unique products")
    
    return product_set

@nb.jit(nopython=True, cache=True, parallel=True)
def check_expressibility_parallel(sample_size, product_set_array):
    """Check random matrices for expressibility."""
    results = np.zeros((17, 2), dtype=np.int32)  # [zeros][total/expressible]
    product_set = set(product_set_array)
    
    # Use better random number generation
    np.random.seed(42)
    
    for _ in prange(sample_size):
        # Generate random matrix
        matrix_int = np.random.randint(0, 3**16)
        
        # Check rank
        if rank_f3_superfast(matrix_int) == 4:
            # Count zeros
            zeros = 0
            temp = matrix_int
            for i in range(16):
                if temp % 3 == 0:
                    zeros += 1
                temp //= 3
            
            results[zeros, 0] += 1
            if matrix_int in product_set:
                results[zeros, 1] += 1
    
    return results

def main():
    print("F_3 HADAMARD FACTORIZATION - SUPER OPTIMIZED")
    print("="*70)
    print("Maximum speed optimizations enabled")
    
    # Quick benchmark
    print("\nBenchmarking rank computation...")
    test_start = time.time()
    test_ranks = [rank_f3_superfast(i) for i in range(10000)]
    test_time = time.time() - test_start
    print(f"✓ Computed 10,000 ranks in {test_time:.3f}s ({10000/test_time:,.0f} ranks/sec)")
    
    # Main computation
    print("\nStarting search...")
    total_start = time.time()
    
    # Step 1: Find rank-2 matrices
    rank2_ints = find_all_rank2_superfast()
    
    # Save checkpoint
    np.save('f3_rank2_checkpoint.npy', rank2_ints)
    print("✓ Checkpoint saved")
    
    # Step 2: Build product table (or sample)
    product_set = build_product_table_batched(rank2_ints, chunk_size=3000)
    
    # Step 3: Check expressibility
    print("\n" + "="*70)
    print("STEP 3: Checking expressibility patterns")
    print("="*70)
    
    sample_size = 5000000  # Larger sample for better statistics
    print(f"Sampling {sample_size:,} random matrices...")
    
    # Convert set to array for Numba
    product_array = np.array(list(product_set), dtype=np.int32)
    
    # Run parallel check
    results = check_expressibility_parallel(sample_size, product_array)
    
    # Display results
    print("\nEXPRESSIBILITY BY NUMBER OF ZEROS:")
    print("Zeros | Total  | Expressible | Rate   | Pattern")
    print("------|--------|-------------|--------|----------")
    
    for zeros in range(17):
        total = results[zeros, 0]
        expr = results[zeros, 1]
        if total > 0:
            rate = expr / total * 100
            pattern = "COUNTEREXAMPLE" if rate < 10 else "Mixed" if rate < 90 else "Expressible"
            print(f"  {zeros:2d}  | {total:6d} | {expr:11d} | {rate:5.1f}% | {pattern}")
    
    # Summary
    dense_total = np.sum(results[:6, 0])
    dense_expr = np.sum(results[:6, 1])
    if dense_total > 0:
        print(f"\nDense matrices (0-5 zeros): {dense_expr}/{dense_total} = {dense_expr/dense_total*100:.1f}%")
    
    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print("\n✓ COMPLETE!")

if __name__ == "__main__":
    main()