#!/usr/bin/env python3
"""
F_3 Hadamard search - Optimized GPU version.
Uses batched rank computation and parallel processing.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import time
from collections import defaultdict
from tqdm.auto import tqdm
import pickle
from multiprocessing import Pool, cpu_count

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"CPU cores available: {cpu_count()}")

def rank_f3_numpy_optimized(M_int):
    """Optimized rank computation - work directly with integer representation."""
    # Extract matrix from integer
    M = np.zeros((4, 4), dtype=np.int8)
    for i in range(16):
        M[i // 4, i % 4] = (M_int // (3**i)) % 3
    
    # Use int8 for faster computation
    rank = 0
    for col in range(4):
        # Find pivot
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
                break
    
    return rank

def process_batch_ranks(args):
    """Process a batch of integers to find their ranks - for multiprocessing."""
    start, end = args
    ranks = []
    rank2_ints = []
    
    for n in range(start, end):
        rank = rank_f3_numpy_optimized(n)
        ranks.append(rank)
        if rank == 2:
            rank2_ints.append(n)
    
    return ranks, rank2_ints

@jit
def int_to_matrix_f3_fast(n):
    """Faster matrix conversion using bit manipulation."""
    indices = jnp.arange(16, dtype=jnp.int32)
    powers = 3 ** indices
    digits = (n // powers) % 3
    return digits.reshape(4, 4)

int_to_matrix_batch = vmap(int_to_matrix_f3_fast)

@jit
def matrix_to_int_f3_fast(M):
    """Faster integer conversion."""
    M_flat = M.flatten()
    indices = jnp.arange(16, dtype=jnp.int32)
    powers = 3 ** indices
    return jnp.dot(M_flat, powers)

@jit
def hadamard_product_f3(A, B):
    """Hadamard product over F_3."""
    return (A * B) % 3

# Vectorized Hadamard product for entire batches
hadamard_batch = vmap(vmap(hadamard_product_f3, in_axes=(None, 0)), in_axes=(0, None))

@jit
def count_zeros(M):
    """Count zeros in matrix."""
    return jnp.sum(M == 0)

count_zeros_batch = vmap(count_zeros)

def find_all_rank2_parallel():
    """Find all rank-2 matrices using parallel processing."""
    print("\n" + "="*70)
    print("STEP 1: Finding all rank-2 matrices (parallel processing)")
    print("="*70)
    
    n_total = 3**16
    n_cores = min(cpu_count(), 32)  # Use up to 32 cores
    chunk_size = 1000000 // n_cores  # Smaller chunks for better distribution
    
    print(f"Using {n_cores} CPU cores for rank computation")
    
    # Prepare work chunks
    tasks = []
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        tasks.append((start, end))
    
    rank2_list = []
    rank_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    pbar = tqdm(total=n_total, desc="Computing ranks", unit="matrices", 
                unit_scale=True)
    
    start_time = time.time()
    
    # Process in parallel
    with Pool(n_cores) as pool:
        for ranks, rank2_ints in pool.imap_unordered(process_batch_ranks, tasks):
            # Update counts
            for rank in ranks:
                rank_counts[rank] += 1
            
            # Collect rank-2 matrices
            rank2_list.extend(rank2_ints)
            
            # Update progress
            pbar.update(len(ranks))
            pbar.set_postfix({'rank-2 found': f'{len(rank2_list):,}'})
    
    pbar.close()
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes ({n_total/elapsed:,.0f} matrices/sec)")
    
    print(f"\nRank distribution:")
    for r in range(5):
        pct = rank_counts[r]/n_total*100
        print(f"  Rank {r}: {rank_counts[r]:,} ({pct:.3f}%)")
    
    return np.array(rank2_list)

def build_product_table_gpu_optimized(rank2_ints, chunk_size=2000):
    """Optimized product table building with larger GPU batches."""
    print("\n" + "="*70)
    print("STEP 2: Building product lookup table (GPU optimized)")
    print("="*70)
    
    # Convert all at once
    print("Converting integers to matrices...")
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
    
    # Process in larger chunks for GPU efficiency
    for i in range(0, n, chunk_size):
        chunk_a = rank2_matrices[i:i+chunk_size]
        chunk_a_ints = rank2_ints[i:i+chunk_size]
        
        # Compute all products with remaining matrices
        for j in range(i, n, chunk_size):
            chunk_b = rank2_matrices[j:j+chunk_size]
            chunk_b_ints = rank2_ints[j:j+chunk_size]
            
            # Batch compute all products at once
            # Shape: (len(chunk_a), len(chunk_b), 4, 4)
            if i == j:
                # Same chunk - compute upper triangle only
                products_batch = vmap(
                    lambda idx_a: vmap(
                        lambda idx_b: hadamard_product_f3(chunk_a[idx_a], chunk_b[idx_b])
                    )(jnp.arange(idx_a, len(chunk_b)))
                )(jnp.arange(len(chunk_a)))
                
                # Convert to integers and store
                for ii in range(len(chunk_a)):
                    for jj in range(ii, len(chunk_b)):
                        product = chunk_a[ii] * chunk_b[jj] % 3
                        prod_int = int(matrix_to_int_f3_fast(product))
                        product_map[prod_int].append((
                            int(chunk_a_ints[ii]), 
                            int(chunk_b_ints[jj])
                        ))
                        pairs_processed += 1
            else:
                # Different chunks - compute all pairs
                products_batch = hadamard_batch(chunk_a, chunk_b)
                
                # Convert to integers and store
                for ii in range(len(chunk_a)):
                    for jj in range(len(chunk_b)):
                        product = products_batch[ii, jj]
                        prod_int = int(matrix_to_int_f3_fast(product))
                        product_map[prod_int].append((
                            int(chunk_a_ints[ii]), 
                            int(chunk_b_ints[jj])
                        ))
                        pairs_processed += 1
            
            pbar.update(pairs_processed - pbar.n)
            pbar.set_postfix({'unique products': f'{len(product_map):,}'})
            
            # Periodic memory cleanup
            if len(product_map) % 100000 == 0:
                jax.clear_caches()
    
    pbar.close()
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes ({total_pairs/elapsed:,.0f} pairs/sec)")
    print(f"Found {len(product_map):,} unique products")
    
    return dict(product_map)

def check_expressibility_parallel(product_map, sample_size=2000000):
    """Check expressibility using parallel rank computation."""
    print("\n" + "="*70)
    print("STEP 3: Checking expressibility of rank-4 matrices")
    print("="*70)
    
    results_by_zeros = defaultdict(lambda: {'total': 0, 'expressible': 0})
    
    # Generate random samples
    key = jax.random.PRNGKey(42)
    print("Generating random samples...")
    sample_ints = jax.random.randint(key, (sample_size,), 0, 3**16)
    sample_ints_np = np.array(sample_ints)
    
    # Convert to matrices for zero counting
    print("Converting to matrices...")
    batch_size = 100000
    all_zeros = []
    for i in range(0, sample_size, batch_size):
        batch = sample_ints[i:i+batch_size]
        matrices = int_to_matrix_batch(batch)
        zeros = count_zeros_batch(matrices)
        all_zeros.extend(np.array(zeros))
    
    all_zeros = np.array(all_zeros)
    
    # Parallel rank computation
    print("Computing ranks in parallel...")
    n_cores = min(cpu_count(), 32)
    chunk_size = sample_size // n_cores
    
    tasks = []
    for i in range(0, sample_size, chunk_size):
        end = min(i + chunk_size, sample_size)
        tasks.append((sample_ints_np[i:end], i))
    
    def process_rank_batch(args):
        ints, offset = args
        ranks = []
        for n in ints:
            ranks.append(rank_f3_numpy_optimized(int(n)))
        return ranks, offset
    
    pbar = tqdm(total=sample_size, desc="Computing ranks", unit="matrices")
    rank4_found = 0
    
    with Pool(n_cores) as pool:
        for ranks, offset in pool.imap_unordered(process_rank_batch, tasks):
            for i, rank in enumerate(ranks):
                if rank == 4:
                    idx = offset + i
                    matrix_int = int(sample_ints_np[idx])
                    zero_count = int(all_zeros[idx])
                    
                    results_by_zeros[zero_count]['total'] += 1
                    rank4_found += 1
                    
                    if matrix_int in product_map:
                        results_by_zeros[zero_count]['expressible'] += 1
            
            pbar.update(len(ranks))
            pbar.set_postfix({'rank-4 found': f'{rank4_found:,}'})
    
    pbar.close()
    
    # Display results
    print(f"\nFound {rank4_found:,} rank-4 matrices in sample of {sample_size:,}")
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
    print("F_3 HADAMARD FACTORIZATION - OPTIMIZED GPU VERSION")
    print("="*70)
    print("Using parallel CPU processing for ranks, GPU for matrix ops")
    
    # Quick test
    print("\nQuick validation...")
    test_rank = rank_f3_numpy_optimized(3**16 - 1)  # All 2s
    print(f"✓ All-2s matrix rank: {test_rank} (expected 1)")
    
    # Main computation
    print("\nStarting exhaustive search...")
    total_start = time.time()
    
    # Step 1: Find rank-2 matrices (parallel)
    rank2_ints = find_all_rank2_parallel()
    print(f"\n✓ Found {len(rank2_ints):,} rank-2 matrices")
    print(f"  Theoretical: ~811,200")
    print(f"  Accuracy: {len(rank2_ints)/811200*100:.1f}%")
    
    # Step 2: Build product table (GPU)
    product_map = build_product_table_gpu_optimized(rank2_ints, chunk_size=2000)
    
    # Step 3: Check expressibility (parallel)
    results = check_expressibility_parallel(product_map, sample_size=2000000)
    
    # Save results
    print("\n" + "="*70)
    print("STEP 4: Saving results")
    print("="*70)
    
    with open('f3_rank2_matrices_opt.npy', 'wb') as f:
        np.save(f, rank2_ints)
    print("✓ Saved rank-2 matrices")
    
    with open('f3_product_map_opt.pkl', 'wb') as f:
        pickle.dump(product_map, f)
    print("✓ Saved product map")
    
    with open('f3_results_opt.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("✓ Saved expressibility results")
    
    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print("\n✓ COMPLETE!")

if __name__ == "__main__":
    main()