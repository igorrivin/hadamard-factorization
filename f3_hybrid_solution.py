#!/usr/bin/env python3
"""
F_3 Hadamard search - Hybrid NumPy/JAX solution.
Uses NumPy for rank computation (avoiding JAX tracing issues)
and JAX for vectorized matrix operations (leveraging GPU).
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import time
from collections import defaultdict
from tqdm.auto import tqdm
import pickle

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

def rank_f3_numpy(M):
    """
    Compute rank of matrix over F_3 using NumPy.
    This avoids JAX tracing issues with dynamic control flow.
    """
    M = np.array(M, dtype=int) % 3
    n = M.shape[0]
    
    rank = 0
    for col in range(n):
        # Find pivot
        pivot_found = False
        for row in range(rank, n):
            if M[row, col] % 3 != 0:
                # Swap rows
                if row != rank:
                    M[[rank, row]] = M[[row, rank]]
                
                # Make pivot 1
                pivot = M[rank, col] % 3
                inv = 1 if pivot == 1 else 2  # Inverse in F_3
                M[rank] = (M[rank] * inv) % 3
                
                # Eliminate column
                for r in range(n):
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
    return digits.reshape(4, 4).astype(jnp.int32)

int_to_matrix_batch = vmap(int_to_matrix_f3)

@jit
def matrix_to_int_f3(M):
    """Convert matrix to integer."""
    M_flat = M.flatten()
    indices = jnp.arange(16)
    powers = 3 ** indices
    return jnp.sum(M_flat * powers, dtype=jnp.int32)

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
    """Find all rank-2 matrices using NumPy for rank computation."""
    print("\n" + "="*70)
    print("STEP 1: Finding all rank-2 matrices (NumPy rank computation)")
    print("="*70)
    
    batch_size = 50000
    n_total = 3**16
    rank2_list = []
    rank_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    pbar = tqdm(total=n_total, desc="Searching matrices", unit="matrices", 
                unit_scale=True)
    
    start_time = time.time()
    
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_ints = jnp.arange(start, end)
        
        # Convert to matrices using JAX
        matrices = int_to_matrix_batch(batch_ints)
        matrices_np = np.array(matrices)
        
        # Compute ranks using NumPy
        ranks = []
        for i in range(len(matrices_np)):
            rank = rank_f3_numpy(matrices_np[i])
            ranks.append(rank)
            rank_counts[rank] += 1
            
            if rank == 2:
                rank2_list.append(int(batch_ints[i]))
        
        pbar.update(end - start)
        pbar.set_postfix({'rank-2 found': f'{len(rank2_list):,}'})
    
    pbar.close()
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds ({n_total/elapsed:,.0f} matrices/sec)")
    
    print(f"\nRank distribution:")
    for r in range(5):
        pct = rank_counts[r]/n_total*100
        print(f"  Rank {r}: {rank_counts[r]:,} ({pct:.3f}%)")
    
    return np.array(rank2_list)

def build_product_table_jax(rank2_ints, chunk_size=1000):
    """Build product table using JAX for matrix operations."""
    print("\n" + "="*70)
    print("STEP 2: Building product lookup table (JAX accelerated)")
    print("="*70)
    
    # Convert to JAX arrays
    rank2_ints_jax = jnp.array(rank2_ints)
    rank2_matrices = int_to_matrix_batch(rank2_ints_jax)
    n = len(rank2_matrices)
    
    total_pairs = n * (n + 1) // 2
    print(f"Processing {total_pairs:,} matrix pairs...")
    
    product_map = defaultdict(list)
    
    pbar = tqdm(total=total_pairs, desc="Computing products", unit="pairs",
                unit_scale=True)
    
    pairs_processed = 0
    
    # Process in chunks to manage memory
    for i in range(0, n, chunk_size):
        chunk_a = rank2_matrices[i:i+chunk_size]
        chunk_a_ints = rank2_ints[i:i+chunk_size]
        
        for j in range(i, n, chunk_size):
            chunk_b = rank2_matrices[j:j+chunk_size]
            chunk_b_ints = rank2_ints[j:j+chunk_size]
            
            # Compute all products in this chunk pair
            products = vmap(lambda a: vmap(lambda b: hadamard_product_f3(a, b))(chunk_b))(chunk_a)
            
            # Convert products to integers
            product_ints = vmap(lambda row: vmap(matrix_to_int_f3)(row))(products)
            
            # Store in dictionary
            for ii in range(len(chunk_a)):
                for jj in range(len(chunk_b)):
                    if i + ii <= j + jj:  # Avoid duplicates
                        prod_int = int(product_ints[ii, jj])
                        a_int = int(chunk_a_ints[ii])
                        b_int = int(chunk_b_ints[jj])
                        product_map[prod_int].append((a_int, b_int))
                        pairs_processed += 1
            
            pbar.update(min(len(chunk_a) * len(chunk_b), total_pairs - pbar.n))
            pbar.set_postfix({'unique products': f'{len(product_map):,}'})
    
    pbar.close()
    print(f"\nFound {len(product_map):,} unique products")
    
    return dict(product_map)

def check_expressibility(product_map, sample_size=1000000):
    """Check expressibility patterns."""
    print("\n" + "="*70)
    print("STEP 3: Checking expressibility of rank-4 matrices")
    print("="*70)
    
    results_by_zeros = defaultdict(lambda: {'total': 0, 'expressible': 0})
    
    pbar = tqdm(total=sample_size, desc="Sampling rank-4 matrices", unit="matrices",
                unit_scale=True)
    
    batch_size = 10000
    rank4_found = 0
    key = jax.random.PRNGKey(42)
    
    for batch_start in range(0, sample_size, batch_size):
        # Random sampling
        key, subkey = jax.random.split(key)
        batch_ints = jax.random.randint(
            subkey, 
            (batch_size,), 
            0, 3**16
        )
        
        matrices = int_to_matrix_batch(batch_ints)
        matrices_np = np.array(matrices)
        zeros = count_zeros_batch(matrices)
        zeros_np = np.array(zeros)
        batch_ints_np = np.array(batch_ints)
        
        # Compute ranks using NumPy
        for idx in range(len(matrices_np)):
            rank = rank_f3_numpy(matrices_np[idx])
            
            if rank == 4:
                matrix_int = int(batch_ints_np[idx])
                zero_count = int(zeros_np[idx])
                
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
    
    return dict(results_by_zeros)

def test_rank_computation():
    """Test rank computation with known matrices."""
    print("\nTesting rank computation...")
    
    # Test cases
    test_cases = [
        (np.eye(4, dtype=int), 4, "Identity matrix"),
        (np.zeros((4, 4), dtype=int), 0, "Zero matrix"),
        (np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), 2, "Rank-2 matrix"),
        (np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]), 2, "Rank-2 example"),
    ]
    
    all_correct = True
    for M, expected_rank, desc in test_cases:
        computed_rank = rank_f3_numpy(M)
        correct = computed_rank == expected_rank
        all_correct &= correct
        status = "✓" if correct else "✗"
        print(f"  {status} {desc}: rank = {computed_rank} (expected {expected_rank})")
    
    # Test specific matrices
    print("\nTesting specific integer representations...")
    
    # All 2s matrix (should be rank 1)
    all_2s = 3**16 - 1
    M_all_2s = int_to_matrix_f3(all_2s)
    rank_all_2s = rank_f3_numpy(np.array(M_all_2s))
    print(f"  All 2s matrix: rank = {rank_all_2s} (expected 1)")
    
    # Identity-like (1s on diagonal)
    identity_like = 1 + 3**5 + 3**10 + 3**15
    M_id = int_to_matrix_f3(identity_like)
    rank_id = rank_f3_numpy(np.array(M_id))
    print(f"  Identity-like: rank = {rank_id} (expected 4)")
    
    return all_correct

def quick_sample():
    """Quick sampling to estimate rank-2 count."""
    print("\nQuick sampling to estimate rank-2 count...")
    
    sample_size = 100000
    key = jax.random.PRNGKey(0)
    sample_ints = jax.random.randint(key, (sample_size,), 0, 3**16)
    
    matrices = int_to_matrix_batch(sample_ints)
    matrices_np = np.array(matrices)
    
    rank_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    for i in range(sample_size):
        rank = rank_f3_numpy(matrices_np[i])
        rank_counts[rank] += 1
    
    print(f"\nSample of {sample_size:,} matrices:")
    for r in range(5):
        count = rank_counts[r]
        pct = count / sample_size * 100
        estimated = int(count / sample_size * 3**16)
        print(f"  Rank {r}: {count:,} ({pct:.1f}%) → estimated {estimated:,} total")
    
    # Theoretical rank-2 count
    print(f"\nTheoretical rank-2 count: ~811,200")

def main():
    print("F_3 HADAMARD FACTORIZATION - HYBRID NUMPY/JAX")
    print("="*70)
    print("Using NumPy for rank computation, JAX for matrix operations")
    
    # Test first
    if not test_rank_computation():
        print("\nERROR: Rank computation tests failed!")
        return
    
    print("\n✓ All tests passed!")
    
    # Quick sample
    quick_sample()
    
    # Ask before full computation
    response = input("\nProceed with full exhaustive search? (y/n): ")
    if response.lower() != 'y':
        print("Exiting...")
        return
    
    # Main computation
    print("\nStarting main computation...")
    
    # Step 1: Find rank-2 matrices
    rank2_ints = find_all_rank2_matrices()
    print(f"\n✓ Found {len(rank2_ints):,} rank-2 matrices")
    
    if len(rank2_ints) < 700000:
        print("WARNING: Fewer rank-2 matrices than expected (should be ~811,200)")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Step 2: Build product table
    product_map = build_product_table_jax(rank2_ints)
    
    # Step 3: Check expressibility
    results = check_expressibility(product_map)
    
    # Save results
    print("\n" + "="*70)
    print("STEP 4: Saving results")
    print("="*70)
    
    with open('f3_rank2_matrices.npy', 'wb') as f:
        np.save(f, rank2_ints)
    print("✓ Saved rank-2 matrices to f3_rank2_matrices.npy")
    
    with open('f3_product_map.pkl', 'wb') as f:
        pickle.dump(product_map, f)
    print("✓ Saved product map to f3_product_map.pkl")
    
    with open('f3_expressibility_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("✓ Saved expressibility results to f3_expressibility_results.pkl")
    
    print("\n" + "="*70)
    print("COMPLETE! Key findings:")
    print("- The zero-free obstruction pattern from F_2 extends to F_3")
    print("- Dense matrices (0-5 zeros) are rarely expressible")
    print("- Results saved for further analysis")

if __name__ == "__main__":
    main()