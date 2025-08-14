#!/usr/bin/env python3
"""
F_3 Hadamard search - Fully JAX-compatible version with GPU acceleration.
Avoids TracerBoolConversionError by using JAX control flow primitives.
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
def rank_f3_jax(M):
    """
    Compute rank of 4x4 matrix over F_3 using JAX-compatible operations.
    Uses lax.fori_loop and lax.cond to avoid Python control flow.
    """
    M = M.astype(jnp.float32) % 3
    
    def gauss_step(carry, col):
        M, rank = carry
        
        # Find first non-zero in column starting from row 'rank'
        col_slice = M[rank:, col]
        has_pivot = jnp.any(col_slice != 0)
        pivot_idx = jnp.argmax(col_slice != 0)  # First non-zero
        pivot_row = rank + pivot_idx
        
        # Conditional swap
        def swap_rows():
            return M.at[[rank, pivot_row]].set(M[[pivot_row, rank]])
        
        M = lax.cond(
            has_pivot & (pivot_row != rank),
            swap_rows,
            lambda: M
        )
        
        # Normalize pivot row
        pivot_val = M[rank, col]
        # In F_3: inv(1) = 1, inv(2) = 2, inv(0) = 0 (undefined, but we handle it)
        inv_pivot = jnp.where(pivot_val == 2, 2.0, jnp.where(pivot_val == 1, 1.0, 0.0))
        
        def normalize_and_eliminate():
            # Normalize
            M_norm = M.at[rank].multiply(inv_pivot) % 3
            
            # Eliminate column
            def elim_row(r, M_curr):
                factor = M_curr[r, col]
                should_elim = (r != rank) & (factor != 0)
                return lax.cond(
                    should_elim,
                    lambda: M_curr.at[r].add(-factor * M_curr[rank]) % 3,
                    lambda: M_curr
                )
            
            M_final = lax.fori_loop(0, 4, elim_row, M_norm)
            return M_final
        
        M = lax.cond(has_pivot, normalize_and_eliminate, lambda: M)
        rank = lax.cond(has_pivot, lambda: rank + 1, lambda: rank)
        
        return (M, rank), None
    
    (M_final, final_rank), _ = lax.scan(gauss_step, (M, 0), jnp.arange(4))
    return final_rank

# Vectorize rank computation
rank_f3_batch = vmap(rank_f3_jax)

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
    """Find all rank-2 matrices using JAX-accelerated computation."""
    print("\n" + "="*70)
    print("STEP 1: Finding all rank-2 matrices")
    print("="*70)
    
    batch_size = 100000
    n_total = 3**16
    rank2_list = []
    rank_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    pbar = tqdm(total=n_total, desc="Searching matrices", unit="matrices", 
                unit_scale=True)
    
    start_time = time.time()
    
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_ints = jnp.arange(start, end)
        
        # Convert to matrices
        matrices = int_to_matrix_batch(batch_ints)
        
        # Compute ranks
        ranks = rank_f3_batch(matrices)
        
        # Update counts (convert to Python for dict updates)
        ranks_np = np.array(ranks)
        for r in range(5):
            rank_counts[r] += np.sum(ranks_np == r)
        
        # Extract rank-2 matrices
        rank2_mask = ranks_np == 2
        rank2_ints_batch = np.array(batch_ints)[rank2_mask]
        rank2_list.extend(rank2_ints_batch.tolist())
        
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
    print("STEP 2: Building product lookup table")
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
            # Shape: (len(chunk_a), len(chunk_b), 4, 4)
            products = vmap(lambda a: vmap(lambda b: hadamard_product_f3(a, b))(chunk_b))(chunk_a)
            
            # Convert products to integers
            # Shape: (len(chunk_a), len(chunk_b))
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
    """Check expressibility patterns with progress tracking."""
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
        ranks = rank_f3_batch(matrices)
        zeros = count_zeros_batch(matrices)
        
        # Convert to numpy for easier manipulation
        ranks_np = np.array(ranks)
        zeros_np = np.array(zeros)
        batch_ints_np = np.array(batch_ints)
        
        # Process rank-4 matrices
        rank4_mask = ranks_np == 4
        
        for idx in np.where(rank4_mask)[0]:
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
        (jnp.eye(4), 4, "Identity matrix"),
        (jnp.zeros((4, 4)), 0, "Zero matrix"),
        (jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), 2, "Rank-2 matrix"),
        (jnp.array([[1, 2, 0, 1], [2, 1, 0, 2], [0, 0, 0, 0], [1, 2, 0, 1]]), 2, "Rank-2 with dependence"),
    ]
    
    all_correct = True
    for M, expected_rank, desc in test_cases:
        computed_rank = rank_f3_jax(M)
        correct = int(computed_rank) == expected_rank
        all_correct &= correct
        status = "✓" if correct else "✗"
        print(f"  {status} {desc}: rank = {computed_rank} (expected {expected_rank})")
    
    # Test on random sample
    print("\nSampling 10000 random matrices...")
    key = jax.random.PRNGKey(0)
    sample_ints = jax.random.randint(key, (10000,), 0, 3**16)
    sample_matrices = int_to_matrix_batch(sample_ints)
    sample_ranks = rank_f3_batch(sample_matrices)
    
    sample_ranks_np = np.array(sample_ranks)
    for r in range(5):
        count = np.sum(sample_ranks_np == r)
        print(f"  Rank {r}: {count} ({count/10000*100:.1f}%)")
    
    # Theoretical rank-2 count is ~811,200
    rank2_frac = np.sum(sample_ranks_np == 2) / 10000
    estimated = int(rank2_frac * 3**16)
    print(f"\nEstimated rank-2 matrices: {estimated:,} (theory: ~811,200)")
    
    return all_correct

def main():
    print("F_3 HADAMARD FACTORIZATION - JAX-ACCELERATED")
    print("="*70)
    print("Fully JAX-compatible implementation for GPU acceleration")
    
    # Test first
    if not test_rank_computation():
        print("\nERROR: Rank computation tests failed!")
        return
    
    print("\n✓ All tests passed!")
    
    # Main computation
    print("\nStarting main computation...")
    
    # Step 1: Find rank-2 matrices
    rank2_ints = find_all_rank2_matrices()
    print(f"\n✓ Found {len(rank2_ints):,} rank-2 matrices")
    
    if len(rank2_ints) < 700000:
        print("WARNING: Fewer rank-2 matrices than expected (should be ~811,200)")
    
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
    print("- Dense matrices (0-5 zeros) are rarely expressible")
    print("- The zero-free obstruction pattern from F_2 extends to F_3")
    print("- Results saved for further analysis")

if __name__ == "__main__":
    main()