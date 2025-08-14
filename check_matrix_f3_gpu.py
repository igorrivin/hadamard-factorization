#!/usr/bin/env python3
"""
GPU-accelerated F3 Hadamard factorization checker using JAX.
Incorporates all optimizations from the CPU version with massive GPU parallelization.
"""

import numpy as np
import sys
import time
import pickle
from typing import Tuple, Optional, Dict

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, device_put
    import jax.lax as lax
    HAS_JAX = True
    
    # Check for GPU
    devices = jax.devices()
    HAS_GPU = any('gpu' in str(d).lower() for d in devices)
    if HAS_GPU:
        print(f"JAX GPU backend available: {devices}")
    else:
        print("JAX CPU backend will be used")
except ImportError:
    HAS_JAX = False
    HAS_GPU = False
    print("JAX not available. Install with: pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    print("For GPU support: pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")

def mod3(x):
    """Convert to {-1, 0, 1} representation."""
    x = x % 3
    return np.where(x == 2, -1, x)

def rank_f3_numpy(M):
    """Compute rank of matrix over F_3 using numpy."""
    A = mod3(M.copy())
    n = A.shape[0]
    rank = 0
    
    for col in range(n):
        found = False
        for row in range(rank, n):
            if A[row, col] != 0:
                if row != rank:
                    A[[rank, row]] = A[[row, rank]]
                if A[rank, col] == -1:
                    A[rank] = -A[rank]
                for r in range(n):
                    if r != rank and A[r, col] != 0:
                        A[r] = mod3(A[r] - A[r, col] * A[rank])
                rank += 1
                found = True
                break
    return rank

def matrix_to_tuple(M):
    """Convert matrix to tuple for hashing."""
    return tuple(M.flatten())

if HAS_JAX:
    @jit
    def matrix_to_int64_jax(M):
        """
        Encode a 4x4 matrix with values in {-1,0,1} as a 64-bit integer.
        Each element uses 2 bits: 00 for 0, 01 for 1, 11 for -1
        """
        result = jnp.int64(0)
        shift = 0
        for i in range(4):
            for j in range(4):
                val = M[i, j]
                bits = jnp.where(val == 1, 1, jnp.where(val == -1, 3, 0))
                result = result | (jnp.int64(bits) << shift)
                shift += 2
        return result
    
    @jit
    def can_divide_batch(M, A_batch):
        """
        Check if M can be divided by each matrix in A_batch.
        Returns (valid_mask, quotient_batch)
        """
        # Expand M to match batch dimension
        M_expanded = jnp.expand_dims(M, 0)  # Shape: (1, 4, 4)
        
        # Check zero compatibility: A[i,j] = 0 implies M[i,j] = 0
        zero_mask = (A_batch == 0)  # Where A is zero
        m_nonzero = (M_expanded != 0)  # Where M is non-zero
        
        # Invalid if A is zero but M is non-zero
        invalid_divisions = jnp.logical_and(zero_mask, m_nonzero)
        valid_mask = ~jnp.any(invalid_divisions, axis=(1, 2))  # Shape: (batch_size,)
        
        # Compute quotients: B = M / A (element-wise)
        # In {-1,0,1}: x/y = x*y for y != 0
        quotients = jnp.where(A_batch != 0, M_expanded * A_batch, 0)
        
        return valid_mask, quotients
    
    @jit
    def encode_matrices_batch(matrices):
        """Encode a batch of matrices to int64."""
        return vmap(matrix_to_int64_jax)(matrices)
    
    @jit 
    def check_factorization_gpu_batch(M, rank2_array, rank2_encoded_set):
        """
        GPU-accelerated factorization check.
        Process large batches in parallel on GPU.
        """
        batch_size = 10000 if HAS_GPU else 1000
        n_matrices = rank2_array.shape[0]
        
        # Process in batches to avoid memory issues
        found_indices = []
        
        for start_idx in range(0, n_matrices, batch_size):
            end_idx = min(start_idx + batch_size, n_matrices)
            A_batch = rank2_array[start_idx:end_idx]
            
            # Check which divisions are valid and compute quotients
            valid_mask, quotients = can_divide_batch(M, A_batch)
            
            # Encode quotients to check if they're rank-2
            quotient_encodings = encode_matrices_batch(quotients)
            
            # Check membership in rank-2 set
            # This is tricky in JAX - we'll check against all rank-2 encodings
            # In practice, you might want to use a different approach
            for i in range(len(quotient_encodings)):
                if valid_mask[i]:
                    # Check if this quotient encoding is in our rank-2 set
                    is_rank2 = jnp.any(rank2_encoded_set == quotient_encodings[i])
                    if is_rank2:
                        return True, start_idx + i, quotients[i]
        
        return False, -1, None

def check_expressibility_f3_gpu(M, rank2_matrices: Dict[tuple, np.ndarray]):
    """GPU-accelerated check for F3 Hadamard factorization."""
    if not HAS_JAX:
        print("JAX not available. Falling back to CPU version...")
        from check_matrix_f3_optimized import check_expressibility_f3_optimized
        return check_expressibility_f3_optimized(M, rank2_matrices)
    
    M = mod3(M).astype(np.int8)
    
    print(f"Checking matrix over F3 (GPU-accelerated):")
    print(M)
    
    m_rank = rank_f3_numpy(M)
    print(f"\nRank: {m_rank}")
    
    if m_rank <= 2:
        print("Matrix has rank ≤ 2, so it's trivially expressible.")
        return True, (M, np.ones((4, 4), dtype=int))
    
    print("\nSearching for factorization on GPU...")
    
    # Convert to JAX arrays and put on GPU
    rank2_list = list(rank2_matrices.values())
    rank2_array = np.array(rank2_list, dtype=np.int8)
    
    # Transfer to GPU
    print(f"Transferring {len(rank2_array)} matrices to GPU...")
    M_jax = device_put(jnp.array(M, dtype=jnp.int8))
    rank2_array_jax = device_put(jnp.array(rank2_array, dtype=jnp.int8))
    
    # Process in large GPU batches
    batch_size = 50000 if HAS_GPU else 5000
    checked = 0
    
    for start_idx in range(0, len(rank2_array), batch_size):
        end_idx = min(start_idx + batch_size, len(rank2_array))
        A_batch = rank2_array_jax[start_idx:end_idx]
        
        # Check divisions for entire batch
        valid_mask, quotients = can_divide_batch(M_jax, A_batch)
        
        # Convert back to numpy for dictionary lookup
        valid_mask_np = np.array(valid_mask)
        quotients_np = np.array(quotients)
        
        # Check each valid quotient
        for i in range(len(valid_mask_np)):
            if valid_mask_np[i]:
                b_tuple = matrix_to_tuple(quotients_np[i])
                if b_tuple in rank2_matrices:
                    A = rank2_array[start_idx + i]
                    B = quotients_np[i]
                    
                    print(f"\nFound factorization!")
                    print(f"A = \n{A}")
                    print(f"B = \n{B}")
                    
                    # Verify
                    product = A * B
                    if np.array_equal(product, M):
                        print("Verification: A ∘ B = M ✓")
                        return True, (A, B)
                    else:
                        print("ERROR: Verification failed!")
                        return False, None
        
        checked += (end_idx - start_idx)
        if checked % 100000 == 0:
            print(f"  Checked {checked}/{len(rank2_array)} matrices...")
    
    print("\nNo factorization found.")
    print("This matrix is a COUNTEREXAMPLE for F3")
    return False, None

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--benchmark':
            if not HAS_JAX:
                print("JAX not installed. Cannot run GPU benchmark.")
                return
                
            print("Benchmarking GPU version...")
            
            # Load precomputed matrices
            try:
                with open('rank2_matrices_f3.pkl', 'rb') as f:
                    rank2_matrices = pickle.load(f)
                print(f"Loaded {len(rank2_matrices)} rank-2 matrices")
            except FileNotFoundError:
                print("Please run check_matrix_f3.py first to generate matrices")
                return
            
            # Test with identity matrix
            test_matrix = np.eye(4, dtype=int)
            
            # Warm up JAX
            print("Warming up JAX...")
            small_test = mod3(test_matrix).astype(np.int8)
            small_batch = list(rank2_matrices.values())[:100]
            M_jax = device_put(jnp.array(small_test))
            A_batch_jax = device_put(jnp.array(small_batch, dtype=jnp.int8))
            _ = can_divide_batch(M_jax, A_batch_jax)
            
            # Benchmark
            print("Running benchmark...")
            start = time.time()
            is_expressible, _ = check_expressibility_f3_gpu(test_matrix, rank2_matrices)
            gpu_time = time.time() - start
            
            print(f"\nResults:")
            print(f"GPU search time: {gpu_time:.3f}s")
            print(f"Device: {'GPU' if HAS_GPU else 'CPU (JAX)'}")
            print(f"Found factorization: {is_expressible}")
            
            # Compare with CPU version
            try:
                from check_matrix_f3_optimized import check_expressibility_f3_optimized
                print("\nComparing with CPU version...")
                start = time.time()
                _, _ = check_expressibility_f3_optimized(test_matrix, rank2_matrices)
                cpu_time = time.time() - start
                print(f"CPU time: {cpu_time:.3f}s")
                print(f"Speedup: {cpu_time/gpu_time:.1f}x")
            except:
                pass
            
            return
    
    # Load precomputed matrices
    try:
        print("Loading precomputed rank-2 matrices...")
        with open('rank2_matrices_f3.pkl', 'rb') as f:
            rank2_matrices = pickle.load(f)
        print(f"Loaded {len(rank2_matrices)} rank-2 matrices")
    except FileNotFoundError:
        print("Error: rank2_matrices_f3.pkl not found.")
        print("Please run: python check_matrix_f3.py --generate-only")
        return
    
    print("\nEnter a 4×4 matrix over F3 (4 lines, 4 values each):")
    print("Values should be 0, 1, or 2")
    
    M = np.zeros((4, 4), dtype=int)
    try:
        for i in range(4):
            line = input().strip().split()
            if len(line) != 4:
                print(f"Error: Row {i+1} must have exactly 4 values")
                return
            for j in range(4):
                val = int(line[j])
                if val not in [0, 1, 2]:
                    print(f"Error: Values must be 0, 1, or 2, got {val}")
                    return
                M[i, j] = val
    except (ValueError, EOFError) as e:
        print(f"Error reading matrix: {e}")
        return
    
    print()
    start_time = time.time()
    is_expressible, factorization = check_expressibility_f3_gpu(M, rank2_matrices)
    print(f"\nSearch took {time.time() - start_time:.3f} seconds")
    
    if is_expressible:
        print("\nResult: This matrix is EXPRESSIBLE as A ∘ B with rank(A) = rank(B) = 2")
    else:
        print("\nResult: This matrix is a COUNTEREXAMPLE")
        print("It cannot be expressed as A ∘ B with rank(A), rank(B) ≤ 2 over F3")

if __name__ == "__main__":
    main()