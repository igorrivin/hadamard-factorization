#!/usr/bin/env python3
"""
F_3 Hadamard search - Numba JIT-compiled version.
Uses Numba to compile the rank computation to machine code.
Should be 10-100x faster than pure NumPy.
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

print("Setting up Numba JIT compilation...")

@nb.jit(nopython=True, cache=True)
def rank_f3_numba(M_int):
    """Ultra-fast rank computation using Numba JIT compilation."""
    # Extract matrix from integer
    M = np.zeros((4, 4), dtype=np.int32)
    for i in range(16):
        M[i // 4, i % 4] = (M_int // (3**i)) % 3
    
    rank = 0
    for col in range(4):
        # Find pivot
        pivot_found = False
        for row in range(rank, 4):
            if M[row, col] % 3 != 0:
                # Swap rows
                if row != rank:
                    for c in range(4):
                        M[rank, c], M[row, c] = M[row, c], M[rank, c]
                
                # Make pivot 1
                pivot = M[rank, col] % 3
                inv = 1 if pivot == 1 else 2
                for c in range(4):
                    M[rank, c] = (M[rank, c] * inv) % 3
                
                # Eliminate column
                for r in range(4):
                    if r != rank and M[r, col] % 3 != 0:
                        factor = M[r, col] % 3
                        for c in range(4):
                            M[r, c] = (M[r, c] - factor * M[rank, c]) % 3
                
                rank += 1
                pivot_found = True
                break
        
        if not pivot_found:
            continue
    
    return rank

@nb.jit(nopython=True, parallel=True, cache=True)
def find_rank2_matrices_numba(start, end):
    """Find all rank-2 matrices in range using parallel Numba."""
    rank2_list = []
    rank_counts = np.zeros(5, dtype=np.int64)
    
    for n in prange(start, end):
        rank = rank_f3_numba(n)
        rank_counts[rank] += 1
        if rank == 2:
            rank2_list.append(n)
    
    return rank2_list, rank_counts

@nb.jit(nopython=True, cache=True)
def matrix_multiply_f3(A_int, B_int):
    """Hadamard product of two matrices given as integers."""
    result = 0
    for i in range(16):
        a_digit = (A_int // (3**i)) % 3
        b_digit = (B_int // (3**i)) % 3
        result += ((a_digit * b_digit) % 3) * (3**i)
    return result

# JAX functions for GPU operations
@jax.jit
def int_to_matrix_f3(n):
    """Convert integer to 4x4 matrix over F_3."""
    indices = jnp.arange(16)
    powers = 3 ** indices
    digits = (n // powers) % 3
    return digits.reshape(4, 4)

int_to_matrix_batch = vmap(int_to_matrix_f3)

@jax.jit
def matrix_to_int_f3(M):
    """Convert matrix to integer."""
    M_flat = M.flatten()
    indices = jnp.arange(16)
    powers = 3 ** indices
    return jnp.sum(M_flat * powers)

@jax.jit
def hadamard_product_f3(A, B):
    """Hadamard product over F_3."""
    return (A * B) % 3

@jax.jit
def count_zeros(M):
    """Count zeros in matrix."""
    return jnp.sum(M == 0)

count_zeros_batch = vmap(count_zeros)

def find_all_rank2_numba():
    """Find all rank-2 matrices using Numba acceleration."""
    print("\n" + "="*70)
    print("STEP 1: Finding all rank-2 matrices (Numba JIT compiled)")
    print("="*70)
    
    n_total = 3**16
    batch_size = 1000000
    rank2_list = []
    total_rank_counts = np.zeros(5, dtype=np.int64)
    
    # Warm up JIT
    print("Warming up JIT compiler...")
    _, _ = find_rank2_matrices_numba(0, 1000)
    
    pbar = tqdm(total=n_total, desc="Searching matrices", unit="matrices", 
                unit_scale=True)
    
    start_time = time.time()
    
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        
        # Use Numba parallel processing
        batch_rank2, batch_counts = find_rank2_matrices_numba(start, end)
        
        rank2_list.extend(batch_rank2)
        total_rank_counts += batch_counts
        
        pbar.update(end - start)
        pbar.set_postfix({'rank-2 found': f'{len(rank2_list):,}'})
        
        # Periodic progress report
        if start > 0 and start % (n_total // 10) == 0:
            elapsed = time.time() - start_time
            rate = start / elapsed
            eta = (n_total - start) / rate
            matrices_per_sec = rate
            print(f"\n  {start/n_total*100:.0f}% complete")
            print(f"  Rate: {matrices_per_sec:,.0f} matrices/sec")
            print(f"  ETA: {eta/60:.1f} minutes")
    
    pbar.close()
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes ({n_total/elapsed:,.0f} matrices/sec)")
    
    print(f"\nRank distribution:")
    for r in range(5):
        pct = total_rank_counts[r]/n_total*100
        print(f"  Rank {r}: {total_rank_counts[r]:,} ({pct:.3f}%)")
    
    return np.array(rank2_list)

def build_product_table_mixed(rank2_ints, chunk_size=2000):
    """Build product table using JAX for GPU acceleration."""
    print("\n" + "="*70)
    print("STEP 2: Building product lookup table")
    print("="*70)
    
    # Convert to matrices using JAX
    print("Converting integers to matrices on GPU...")
    rank2_ints_jax = jnp.array(rank2_ints)
    rank2_matrices = int_to_matrix_batch(rank2_ints_jax)
    n = len(rank2_matrices)
    
    total_pairs = n * (n + 1) // 2
    print(f"Processing {total_pairs:,} matrix pairs...")
    
    product_map = defaultdict(list)
    
    pbar = tqdm(total=total_pairs, desc="Computing products", unit="pairs",
                unit_scale=True)
    
    start_time = time.time()
    pairs_processed = 0
    
    # Process in chunks
    for i in range(0, n, chunk_size):
        chunk_a = rank2_matrices[i:i+chunk_size]
        chunk_a_ints = rank2_ints[i:i+chunk_size]
        
        for j in range(i, n, chunk_size):
            chunk_b = rank2_matrices[j:j+chunk_size]
            chunk_b_ints = rank2_ints[j:j+chunk_size]
            
            # Compute products using JAX
            products = vmap(lambda a: vmap(lambda b: hadamard_product_f3(a, b))(chunk_b))(chunk_a)
            
            # Convert back to integers
            product_ints = vmap(lambda row: vmap(matrix_to_int_f3)(row))(products)
            
            # Store results
            for ii in range(len(chunk_a)):
                for jj in range(len(chunk_b)):
                    if i + ii <= j + jj:
                        prod_int = int(product_ints[ii, jj])
                        a_int = int(chunk_a_ints[ii])
                        b_int = int(chunk_b_ints[jj])
                        product_map[prod_int].append((a_int, b_int))
                        pairs_processed += 1
            
            pbar.update(pairs_processed - pbar.n)
            pbar.set_postfix({'unique products': f'{len(product_map):,}'})
    
    pbar.close()
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes ({total_pairs/elapsed:,.0f} pairs/sec)")
    print(f"Found {len(product_map):,} unique products")
    
    return dict(product_map)

@nb.jit(nopython=True, parallel=True, cache=True)
def check_ranks_numba(sample_ints):
    """Check ranks of sample matrices using parallel Numba."""
    n = len(sample_ints)
    ranks = np.zeros(n, dtype=np.int32)
    
    for i in prange(n):
        ranks[i] = rank_f3_numba(sample_ints[i])
    
    return ranks

def check_expressibility_numba(product_map, sample_size=2000000):
    """Check expressibility using Numba-accelerated rank computation."""
    print("\n" + "="*70)
    print("STEP 3: Checking expressibility of rank-4 matrices")
    print("="*70)
    
    results_by_zeros = defaultdict(lambda: {'total': 0, 'expressible': 0})
    
    # Generate random samples
    key = jax.random.PRNGKey(42)
    print("Generating random samples...")
    sample_ints = jax.random.randint(key, (sample_size,), 0, 3**16)
    sample_ints_np = np.array(sample_ints, dtype=np.int64)
    
    # Count zeros using JAX
    print("Counting zeros in matrices...")
    matrices = int_to_matrix_batch(sample_ints)
    zeros = count_zeros_batch(matrices)
    zeros_np = np.array(zeros)
    
    # Compute ranks using Numba
    print("Computing ranks with Numba (parallel)...")
    ranks = check_ranks_numba(sample_ints_np)
    
    # Process results
    print("Analyzing expressibility...")
    rank4_found = 0
    
    pbar = tqdm(total=sample_size, desc="Processing results")
    
    for i in range(sample_size):
        if ranks[i] == 4:
            matrix_int = int(sample_ints_np[i])
            zero_count = int(zeros_np[i])
            
            results_by_zeros[zero_count]['total'] += 1
            rank4_found += 1
            
            if matrix_int in product_map:
                results_by_zeros[zero_count]['expressible'] += 1
        
        if i % 10000 == 0:
            pbar.update(10000)
            pbar.set_postfix({'rank-4 found': f'{rank4_found:,}'})
    
    pbar.close()
    
    # Display results
    print(f"\nFound {rank4_found:,} rank-4 matrices in sample")
    print("\n" + "="*70)
    print("EXPRESSIBILITY BY NUMBER OF ZEROS:")
    print("="*70)
    print("Zeros | Total  | Expressible | Rate   | Pattern")
    print("------|--------|-------------|--------|----------")
    
    for zeros in sorted(results_by_zeros.keys()):
        r = results_by_zeros[zeros]
        if r['total'] > 0:
            rate = r['expressible'] / r['total'] * 100
            pattern = "COUNTEREXAMPLE" if rate < 10 else "Mixed" if rate < 90 else "Expressible"
            print(f"  {zeros:2d}  | {r['total']:6d} | {r['expressible']:11d} | {rate:5.1f}% | {pattern}")
    
    return dict(results_by_zeros)

def main():
    print("F_3 HADAMARD FACTORIZATION - NUMBA ACCELERATED")
    print("="*70)
    print("Using Numba JIT compilation for massive speedup")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    
    # Test Numba compilation
    print("\nTesting Numba JIT compilation...")
    test_start = time.time()
    test_rank = rank_f3_numba(3**16 - 1)
    test_time = time.time() - test_start
    print(f"✓ All-2s matrix rank: {test_rank} (computed in {test_time*1000:.2f}ms)")
    
    # Main computation
    print("\nStarting exhaustive search...")
    total_start = time.time()
    
    # Step 1: Find rank-2 matrices
    rank2_ints = find_all_rank2_numba()
    print(f"\n✓ Found {len(rank2_ints):,} rank-2 matrices")
    print(f"  Theoretical: ~811,200")
    print(f"  Accuracy: {len(rank2_ints)/811200*100:.1f}%")
    
    # Step 2: Build product table
    product_map = build_product_table_mixed(rank2_ints, chunk_size=2000)
    
    # Step 3: Check expressibility
    results = check_expressibility_numba(product_map, sample_size=2000000)
    
    # Save results
    print("\n" + "="*70)
    print("STEP 4: Saving results")
    print("="*70)
    
    with open('f3_rank2_matrices_numba.npy', 'wb') as f:
        np.save(f, rank2_ints)
    print("✓ Saved rank-2 matrices")
    
    with open('f3_product_map_numba.pkl', 'wb') as f:
        pickle.dump(product_map, f)
    print("✓ Saved product map")
    
    with open('f3_results_numba.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("✓ Saved expressibility results")
    
    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    
    # Performance summary
    matrices_per_sec = 3**16 / (total_elapsed / 3)  # Approximate for just rank computation
    print(f"\nPerformance: ~{matrices_per_sec:,.0f} matrices/sec for rank computation")
    print("\n✓ COMPLETE!")

if __name__ == "__main__":
    main()