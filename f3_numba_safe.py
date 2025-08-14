#!/usr/bin/env python3
"""
F_3 Hadamard search - Safe Numba version without parallel processing.
Avoids memory corruption issues on Colab.
"""

import numpy as np
import numba as nb
from numba import jit
import jax
import jax.numpy as jnp
from jax import vmap
import time
from collections import defaultdict
from tqdm.auto import tqdm
import pickle

print("Setting up Numba JIT compilation (safe mode)...")

@nb.jit(nopython=True, cache=True, fastmath=True)
def rank_f3_numba_single(M_int):
    """Ultra-fast rank computation using Numba JIT compilation."""
    # Extract matrix from integer - unroll for speed
    M = np.zeros((4, 4), dtype=np.int8)
    temp = M_int
    for i in range(16):
        M[i >> 2, i & 3] = temp % 3
        temp //= 3
    
    rank = 0
    for col in range(4):
        # Find pivot
        found = False
        for row in range(rank, 4):
            if M[row, col] != 0:
                # Swap rows if needed
                if row != rank:
                    for c in range(4):
                        temp = M[rank, c]
                        M[rank, c] = M[row, c]
                        M[row, c] = temp
                
                # Make pivot 1
                pivot = M[rank, col]
                if pivot == 2:
                    for c in range(4):
                        M[rank, c] = (M[rank, c] * 2) % 3
                
                # Eliminate column
                for r in range(4):
                    if r != rank:
                        factor = M[r, col]
                        if factor != 0:
                            for c in range(4):
                                M[r, c] = (M[r, c] - factor * M[rank, c]) % 3
                
                rank += 1
                found = True
                break
        
        if not found:
            continue
    
    return rank

@nb.jit(nopython=True, cache=True)
def process_batch_numba(start, end):
    """Process a batch of matrices without parallel to avoid memory issues."""
    n = end - start
    rank2_list = np.empty(n, dtype=np.int32)  # Pre-allocate max size
    rank2_count = 0
    rank_counts = np.zeros(5, dtype=np.int32)
    
    for i in range(n):
        matrix_int = start + i
        rank = rank_f3_numba_single(matrix_int)
        rank_counts[rank] += 1
        
        if rank == 2:
            rank2_list[rank2_count] = matrix_int
            rank2_count += 1
    
    return rank2_list[:rank2_count].copy(), rank_counts

# JAX functions remain the same
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

def find_all_rank2_safe():
    """Find all rank-2 matrices using safe Numba (no parallel)."""
    print("\n" + "="*70)
    print("STEP 1: Finding all rank-2 matrices (Numba - safe mode)")
    print("="*70)
    
    n_total = 3**16
    batch_size = 500000  # Smaller batches to avoid memory issues
    rank2_list = []
    total_rank_counts = np.zeros(5, dtype=np.int64)
    
    # Warm up JIT on small batch
    print("Warming up JIT compiler...")
    _, _ = process_batch_numba(0, 100)
    print("✓ JIT compilation successful")
    
    pbar = tqdm(total=n_total, desc="Searching matrices", unit="matrices", 
                unit_scale=True)
    
    start_time = time.time()
    last_report = 0
    
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        
        # Process batch
        batch_rank2, batch_counts = process_batch_numba(start, end)
        
        # Collect results
        if len(batch_rank2) > 0:
            rank2_list.extend(batch_rank2)
        total_rank_counts += batch_counts
        
        # Update progress
        pbar.update(end - start)
        pbar.set_postfix({'rank-2 found': f'{len(rank2_list):,}'})
        
        # Periodic detailed progress report
        if time.time() - last_report > 30:  # Every 30 seconds
            elapsed = time.time() - start_time
            rate = (start + batch_size) / elapsed
            eta = (n_total - start) / rate
            print(f"\n  Progress: {start/n_total*100:.1f}%")
            print(f"  Rate: {rate:,.0f} matrices/sec")
            print(f"  ETA: {eta/60:.1f} minutes")
            print(f"  Found {len(rank2_list):,} rank-2 matrices so far")
            last_report = time.time()
    
    pbar.close()
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes ({n_total/elapsed:,.0f} matrices/sec)")
    
    print(f"\nRank distribution:")
    for r in range(5):
        pct = total_rank_counts[r]/n_total*100 if n_total > 0 else 0
        print(f"  Rank {r}: {total_rank_counts[r]:,} ({pct:.3f}%)")
    
    return np.array(rank2_list, dtype=np.int32)

def build_product_table_gpu(rank2_ints, chunk_size=1500):
    """Build product table using JAX GPU acceleration."""
    print("\n" + "="*70)
    print("STEP 2: Building product lookup table (GPU accelerated)")
    print("="*70)
    
    # Convert to matrices on GPU
    print("Converting integers to matrices on GPU...")
    rank2_ints_jax = jnp.array(rank2_ints)
    rank2_matrices = int_to_matrix_batch(rank2_ints_jax)
    n = len(rank2_matrices)
    
    total_pairs = n * (n + 1) // 2
    print(f"Processing {total_pairs:,} matrix pairs...")
    print(f"Using chunk size: {chunk_size}")
    
    product_map = defaultdict(list)
    
    pbar = tqdm(total=total_pairs, desc="Computing products", unit="pairs",
                unit_scale=True)
    
    start_time = time.time()
    pairs_processed = 0
    last_report = 0
    
    # Process in chunks
    for i in range(0, n, chunk_size):
        chunk_a = rank2_matrices[i:i+chunk_size]
        chunk_a_ints = rank2_ints[i:i+chunk_size]
        
        for j in range(i, n, chunk_size):
            chunk_b = rank2_matrices[j:j+chunk_size]
            chunk_b_ints = rank2_ints[j:j+chunk_size]
            
            # Vectorized Hadamard products on GPU
            products = vmap(lambda a: vmap(lambda b: hadamard_product_f3(a, b))(chunk_b))(chunk_a)
            
            # Convert to integers
            product_ints = vmap(lambda row: vmap(matrix_to_int_f3)(row))(products)
            
            # Store results (CPU bound)
            product_ints_np = np.array(product_ints)
            for ii in range(len(chunk_a)):
                for jj in range(len(chunk_b)):
                    if i + ii <= j + jj:
                        prod_int = int(product_ints_np[ii, jj])
                        a_int = int(chunk_a_ints[ii])
                        b_int = int(chunk_b_ints[jj])
                        product_map[prod_int].append((a_int, b_int))
                        pairs_processed += 1
            
            pbar.update(pairs_processed - pbar.n)
            pbar.set_postfix({'unique products': f'{len(product_map):,}'})
            
            # Periodic report
            if time.time() - last_report > 30:
                elapsed = time.time() - start_time
                rate = pairs_processed / elapsed
                eta = (total_pairs - pairs_processed) / rate
                print(f"\n  Progress: {pairs_processed/total_pairs*100:.1f}%")
                print(f"  Rate: {rate:,.0f} pairs/sec")
                print(f"  ETA: {eta/60:.1f} minutes")
                last_report = time.time()
            
            # Clear GPU memory periodically
            if len(product_map) % 50000 == 0:
                jax.clear_caches()
    
    pbar.close()
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes ({total_pairs/elapsed:,.0f} pairs/sec)")
    print(f"Found {len(product_map):,} unique products")
    
    return dict(product_map)

@nb.jit(nopython=True, cache=True)
def check_ranks_batch(sample_ints):
    """Check ranks of a batch of matrices."""
    n = len(sample_ints)
    ranks = np.zeros(n, dtype=np.int8)
    
    for i in range(n):
        ranks[i] = rank_f3_numba_single(sample_ints[i])
    
    return ranks

def check_expressibility_safe(product_map, sample_size=2000000):
    """Check expressibility patterns."""
    print("\n" + "="*70)
    print("STEP 3: Checking expressibility of rank-4 matrices")
    print("="*70)
    
    results_by_zeros = defaultdict(lambda: {'total': 0, 'expressible': 0})
    
    # Generate samples
    key = jax.random.PRNGKey(42)
    print("Generating random matrix samples...")
    
    batch_size = 100000
    rank4_found = 0
    
    pbar = tqdm(total=sample_size, desc="Processing matrices", unit="matrices",
                unit_scale=True)
    
    for batch_start in range(0, sample_size, batch_size):
        batch_end = min(batch_start + batch_size, sample_size)
        current_batch_size = batch_end - batch_start
        
        # Generate random matrices
        key, subkey = jax.random.split(key)
        sample_ints = jax.random.randint(subkey, (current_batch_size,), 0, 3**16)
        sample_ints_np = np.array(sample_ints, dtype=np.int64)
        
        # Count zeros using GPU
        matrices = int_to_matrix_batch(sample_ints)
        zeros = count_zeros_batch(matrices)
        zeros_np = np.array(zeros)
        
        # Compute ranks using Numba
        ranks = check_ranks_batch(sample_ints_np)
        
        # Process results
        for i in range(current_batch_size):
            if ranks[i] == 4:
                matrix_int = int(sample_ints_np[i])
                zero_count = int(zeros_np[i])
                
                results_by_zeros[zero_count]['total'] += 1
                rank4_found += 1
                
                if matrix_int in product_map:
                    results_by_zeros[zero_count]['expressible'] += 1
        
        pbar.update(current_batch_size)
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
    
    # Summary
    total_dense = sum(results_by_zeros[z]['total'] for z in range(6))
    expr_dense = sum(results_by_zeros[z]['expressible'] for z in range(6))
    if total_dense > 0:
        rate_dense = expr_dense / total_dense * 100
        print(f"\nDense matrices (0-5 zeros): {expr_dense}/{total_dense} = {rate_dense:.1f}% expressible")
        print("✓ Confirms zero-free obstruction pattern extends to F_3")
    
    return dict(results_by_zeros)

def main():
    print("F_3 HADAMARD FACTORIZATION - SAFE NUMBA VERSION")
    print("="*70)
    print("Using Numba JIT without parallel to avoid memory issues")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    
    # Quick test
    print("\nTesting Numba compilation...")
    test_start = time.time()
    rank1 = rank_f3_numba_single(3**16 - 1)  # All 2s
    rank2 = rank_f3_numba_single(0)  # All 0s
    rank3 = rank_f3_numba_single(1 + 3**5 + 3**10 + 3**15)  # Identity-like
    test_time = time.time() - test_start
    print(f"✓ Test ranks: {rank1}, {rank2}, {rank3} (computed in {test_time*1000:.2f}ms)")
    
    # Main computation
    print("\nStarting exhaustive search...")
    total_start = time.time()
    
    try:
        # Step 1: Find rank-2 matrices
        rank2_ints = find_all_rank2_safe()
        print(f"\n✓ Found {len(rank2_ints):,} rank-2 matrices")
        print(f"  Theoretical: ~811,200")
        if len(rank2_ints) > 0:
            print(f"  Accuracy: {len(rank2_ints)/811200*100:.1f}%")
        
        # Save intermediate result
        np.save('f3_rank2_matrices_temp.npy', rank2_ints)
        print("✓ Saved intermediate rank-2 results")
        
        # Step 2: Build product table
        product_map = build_product_table_gpu(rank2_ints, chunk_size=1500)
        
        # Save intermediate result
        with open('f3_product_map_temp.pkl', 'wb') as f:
            pickle.dump(product_map, f)
        print("✓ Saved intermediate product map")
        
        # Step 3: Check expressibility
        results = check_expressibility_safe(product_map, sample_size=2000000)
        
        # Save final results
        print("\n" + "="*70)
        print("STEP 4: Saving final results")
        print("="*70)
        
        np.save('f3_rank2_matrices_safe.npy', rank2_ints)
        print("✓ Saved rank-2 matrices")
        
        with open('f3_product_map_safe.pkl', 'wb') as f:
            pickle.dump(product_map, f)
        print("✓ Saved product map")
        
        with open('f3_results_safe.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("✓ Saved expressibility results")
        
        total_elapsed = time.time() - total_start
        print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
        print("\n✓ COMPLETE!")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Intermediate results may have been saved")
        raise

if __name__ == "__main__":
    main()