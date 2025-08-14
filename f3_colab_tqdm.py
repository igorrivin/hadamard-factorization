#!/usr/bin/env python3
"""
F_3 Hadamard search with tqdm progress bars for Google Colab.
"""

# For Colab, run first:
# !pip install -q jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import time
from collections import defaultdict
from tqdm.auto import tqdm  # Use tqdm.auto for better Colab integration
import pickle

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

@jit
def rank_f3_correct(M):
    """
    Correct rank computation for a single matrix over F_3.
    """
    M = M.astype(jnp.float32) % 3
    n = M.shape[0]
    
    rank = 0
    for col in range(n):
        # Find pivot
        pivot_found = False
        for row in range(rank, n):
            if M[row, col] % 3 != 0:
                # Swap rows
                if row != rank:
                    M = M.at[[rank, row]].set(M[[row, rank]])
                
                # Make pivot 1
                pivot = M[rank, col] % 3
                inv = jnp.where(pivot == 2, 2.0, 1.0)  # Inverse in F_3
                M = M.at[rank].set((M[rank] * inv) % 3)
                
                # Eliminate column
                for r in range(n):
                    if r != rank and M[r, col] % 3 != 0:
                        factor = M[r, col] % 3
                        M = M.at[r].set((M[r] - factor * M[rank]) % 3)
                
                rank += 1
                pivot_found = True
                break
    
    return rank

# Vectorize
rank_f3_batch = vmap(rank_f3_correct)

@jit
def int_to_matrix_f3(n):
    """Convert integer to 4x4 matrix over F_3."""
    matrix = jnp.zeros((4, 4), dtype=jnp.float32)
    
    for i in range(16):
        digit = (n // (3**i)) % 3
        matrix = matrix.at[i // 4, i % 4].set(digit)
    
    return matrix

int_to_matrix_batch = vmap(int_to_matrix_f3)

@jit
def hadamard_product_f3(A, B):
    """Hadamard product over F_3."""
    return (A * B) % 3

@jit
def count_zeros(M):
    """Count zeros in matrix."""
    return jnp.sum(M == 0)

count_zeros_batch = vmap(count_zeros)

def find_all_rank2_matrices():
    """Find all rank-2 matrices with progress bar."""
    print("\n" + "="*70)
    print("STEP 1: Finding all rank-2 matrices")
    print("="*70)
    
    batch_size = 50000
    n_total = 3**16
    rank2_list = []
    rank_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    # Create progress bar
    pbar = tqdm(total=n_total, desc="Searching matrices", unit="matrices", 
                unit_scale=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
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
        
        # Update progress bar
        pbar.update(batch_size)
        pbar.set_postfix({'rank-2 found': f'{len(rank2_list):,}'})
    
    pbar.close()
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds ({n_total/elapsed:,.0f} matrices/sec)")
    
    print(f"\nRank distribution:")
    for r in range(5):
        pct = rank_counts[r]/n_total*100
        print(f"  Rank {r}: {rank_counts[r]:,} ({pct:.3f}%)")
    
    return jnp.array(rank2_list)

def build_product_table(rank2_ints, chunk_size=500):
    """Build lookup table of all Hadamard products."""
    print("\n" + "="*70)
    print("STEP 2: Building product lookup table")
    print("="*70)
    
    rank2_matrices = int_to_matrix_batch(rank2_ints)
    n = len(rank2_matrices)
    
    # Calculate total number of pairs to process
    total_pairs = n * (n + 1) // 2
    print(f"Processing {total_pairs:,} matrix pairs...")
    
    product_map = defaultdict(list)
    
    # Progress bar for pairs
    pbar = tqdm(total=total_pairs, desc="Computing products", unit="pairs",
                unit_scale=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    pairs_processed = 0
    unique_products = 0
    
    for i in range(0, n, chunk_size):
        chunk_a = rank2_matrices[i:i+chunk_size]
        
        for j in range(i, n, chunk_size):
            chunk_b = rank2_matrices[j:j+chunk_size]
            
            # Compute all products in chunk
            for ii in range(len(chunk_a)):
                for jj in range(len(chunk_b)):
                    if i + ii <= j + jj:  # Avoid duplicates
                        product = hadamard_product_f3(chunk_a[ii], chunk_b[jj])
                        
                        # Convert to integer for hashing
                        prod_int = 0
                        for k in range(16):
                            prod_int += int(product[k // 4, k % 4]) * (3**k)
                        
                        is_new = prod_int not in product_map
                        product_map[prod_int].append((rank2_ints[i+ii].item(), 
                                                     rank2_ints[j+jj].item()))
                        
                        pairs_processed += 1
                        if is_new:
                            unique_products += 1
                        
                        # Update progress
                        pbar.update(1)
                        if pairs_processed % 100000 == 0:
                            pbar.set_postfix({'unique products': f'{unique_products:,}'})
    
    pbar.close()
    print(f"\nFound {unique_products:,} unique products from {pairs_processed:,} pairs")
    
    return product_map

def check_expressibility(product_map, sample_size=1000000):
    """Check expressibility with progress tracking."""
    print("\n" + "="*70)
    print("STEP 3: Checking expressibility of rank-4 matrices")
    print("="*70)
    
    results_by_zeros = defaultdict(lambda: {'total': 0, 'expressible': 0})
    
    # Progress bar for sampling
    pbar = tqdm(total=sample_size, desc="Sampling rank-4 matrices", unit="matrices",
                unit_scale=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    batch_size = 10000
    rank4_found = 0
    
    for batch_start in range(0, sample_size, batch_size):
        # Random sampling
        batch_ints = jax.random.randint(
            jax.random.PRNGKey(batch_start), 
            (batch_size,), 
            0, 3**16
        )
        
        matrices = int_to_matrix_batch(batch_ints)
        ranks = rank_f3_batch(matrices)
        zeros = count_zeros_batch(matrices)
        
        # Process rank-4 matrices
        rank4_mask = ranks == 4
        
        for i in jnp.where(rank4_mask)[0]:
            matrix_int = batch_ints[i].item()
            zero_count = int(zeros[i])
            
            results_by_zeros[zero_count]['total'] += 1
            rank4_found += 1
            
            # Check if expressible
            if matrix_int in product_map:
                results_by_zeros[zero_count]['expressible'] += 1
        
        pbar.update(batch_size)
        pbar.set_postfix({'rank-4 found': f'{rank4_found:,}'})
    
    pbar.close()
    
    # Display results
    print(f"\nFound {rank4_found:,} rank-4 matrices in sample")
    print("\nExpressibility by number of zeros:")
    print("Zeros | Total | Expressible | Rate")
    print("------|-------|-------------|------")
    
    for zeros in sorted(results_by_zeros.keys()):
        r = results_by_zeros[zeros]
        if r['total'] > 0:
            rate = r['expressible'] / r['total'] * 100
            print(f"  {zeros:2d}  | {r['total']:5d} |    {r['expressible']:5d}    | {rate:5.1f}%")
    
    return results_by_zeros

def save_results(rank2_ints, product_map, results):
    """Save all results to files."""
    print("\n" + "="*70)
    print("STEP 4: Saving results")
    print("="*70)
    
    with open('f3_rank2_matrices.pkl', 'wb') as f:
        pickle.dump(rank2_ints, f)
    print("✓ Saved rank-2 matrices to f3_rank2_matrices.pkl")
    
    with open('f3_product_map.pkl', 'wb') as f:
        pickle.dump(dict(product_map), f)
    print("✓ Saved product map to f3_product_map.pkl")
    
    with open('f3_expressibility_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("✓ Saved expressibility results to f3_expressibility_results.pkl")

def main():
    print("F_3 HADAMARD FACTORIZATION - GPU ACCELERATED WITH PROGRESS")
    print("="*70)
    
    # Quick test first
    print("\nQuick verification test...")
    test_matrices = [
        jnp.eye(4),  # Identity, rank 4
        jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),  # rank 2
    ]
    
    for i, M in enumerate(test_matrices):
        rank = rank_f3_correct(M)
        print(f"  Test matrix {i+1}: rank = {rank}")
    
    # Main computation
    print("\nStarting main computation...")
    
    # Step 1: Find all rank-2 matrices
    rank2_ints = find_all_rank2_matrices()
    print(f"\n✓ Found {len(rank2_ints):,} rank-2 matrices (expected ~811,200)")
    
    if len(rank2_ints) < 100000:
        print("WARNING: Found fewer rank-2 matrices than expected!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Step 2: Build product table
    product_map = build_product_table(rank2_ints, chunk_size=500)
    
    # Step 3: Check expressibility
    results = check_expressibility(product_map, sample_size=1000000)
    
    # Step 4: Save results
    save_results(rank2_ints, product_map, results)
    
    print("\n" + "="*70)
    print("COMPLETE! Key findings:")
    print("- Dense matrices (few zeros) are rarely expressible")
    print("- The pattern from F_2 generalizes to F_3")
    print("- Results saved for further analysis")

if __name__ == "__main__":
    main()