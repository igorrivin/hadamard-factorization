#!/usr/bin/env python3
"""
Analyze the feasibility of extending the Hadamard factorization search to F_3.
"""

import numpy as np
import time
from itertools import product

def matrix_rank_f3(M):
    """Compute rank of matrix over F_3 using Gaussian elimination."""
    M = M.copy() % 3
    n = len(M)
    rank = 0
    
    for col in range(n):
        # Find pivot
        for row in range(rank, n):
            if M[row, col] % 3 != 0:
                # Swap rows
                M[[rank, row]] = M[[row, rank]]
                # Make pivot 1
                inv = pow(int(M[rank, col]), 1, 3)  # Inverse in F_3
                M[rank] = (M[rank] * inv) % 3
                # Eliminate column
                for r in range(n):
                    if r != rank and M[r, col] != 0:
                        M[r] = (M[r] - M[r, col] * M[rank]) % 3
                rank += 1
                break
    
    return rank

def count_matrices_by_rank_f3():
    """Count 4x4 matrices over F_3 by rank (using sampling for estimates)."""
    print("ESTIMATING MATRIX DISTRIBUTION OVER F_3")
    print("="*50)
    
    # For exact counts of small matrices
    print("\nExact counts for 2x2 matrices over F_3:")
    rank_counts_2x2 = {0: 0, 1: 0, 2: 0}
    
    for vals in product(range(3), repeat=4):
        M = np.array(vals).reshape(2, 2)
        rank = matrix_rank_f3(M)
        rank_counts_2x2[rank] += 1
    
    total_2x2 = sum(rank_counts_2x2.values())
    for rank, count in sorted(rank_counts_2x2.items()):
        print(f"  Rank {rank}: {count} ({count/total_2x2*100:.1f}%)")
    
    # Theoretical calculation for 4x4
    print("\nTheoretical estimates for 4x4 matrices over F_3:")
    
    # Rank 4: product of (1 - 3^(-i)) for i = 1,2,3,4
    prob_rank4 = 1
    for i in range(1, 5):
        prob_rank4 *= (1 - 3**(-i))
    
    total_4x4 = 3**16
    rank4_count = int(total_4x4 * prob_rank4)
    
    print(f"  Total matrices: {total_4x4:,}")
    print(f"  Rank 4 (estimated): {rank4_count:,} ({prob_rank4*100:.1f}%)")
    
    # Sample-based estimation for rank 2
    print("\nSampling to estimate rank-2 matrices...")
    n_samples = 100000
    rank_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    start = time.time()
    for _ in range(n_samples):
        # Random 4x4 matrix over F_3
        M = np.random.randint(0, 3, (4, 4))
        rank = matrix_rank_f3(M)
        rank_counts[rank] += 1
    
    elapsed = time.time() - start
    print(f"  Sampled {n_samples:,} matrices in {elapsed:.2f}s")
    
    print("\n  Estimated distribution:")
    for rank, count in sorted(rank_counts.items()):
        est_total = int(total_4x4 * count / n_samples)
        print(f"    Rank {rank}: ~{est_total:,} ({count/n_samples*100:.1f}%)")
    
    rank2_est = int(total_4x4 * rank_counts[2] / n_samples)
    return rank4_count, rank2_est

def analyze_computational_cost(rank4_count, rank2_count):
    """Analyze the computational cost of exhaustive search."""
    print("\n\nCOMPUTATIONAL COST ANALYSIS")
    print("="*50)
    
    print(f"\nProblem size:")
    print(f"  Full-rank matrices to check: ~{rank4_count:,}")
    print(f"  Rank-2 matrices: ~{rank2_count:,}")
    
    # Number of pairs to check
    pairs = rank2_count * (rank2_count + 1) // 2
    print(f"  Pairs of rank-2 matrices: ~{pairs:,}")
    
    # Time estimate
    ops_per_sec = 1e7  # Optimistic estimate
    time_est = pairs / ops_per_sec
    
    print(f"\nTime estimate (at {ops_per_sec:.0e} ops/sec):")
    print(f"  {time_est:.0f} seconds = {time_est/3600:.1f} hours = {time_est/86400:.1f} days")
    
    # Memory requirements
    print(f"\nMemory requirements:")
    print(f"  Storing all rank-2 matrices: ~{rank2_count * 16 / 1e9:.1f} GB")
    print(f"  Storing results: ~{rank4_count * 4 / 1e9:.1f} GB")
    
    return pairs

def test_density_patterns():
    """Test how density patterns work in F_3."""
    print("\n\nDENSITY PATTERNS IN F_3")
    print("="*50)
    
    # In F_3, we have 0, 1, 2 as elements
    # "Density" could be measured as number of non-zero elements
    
    print("\nSampling density distribution of full-rank matrices...")
    n_samples = 10000
    density_rank4 = []
    density_other = []
    
    for _ in range(n_samples):
        M = np.random.randint(0, 3, (4, 4))
        rank = matrix_rank_f3(M)
        nonzeros = np.sum(M != 0)
        
        if rank == 4:
            density_rank4.append(nonzeros)
        else:
            density_other.append(nonzeros)
    
    print(f"  Average non-zeros in rank-4 matrices: {np.mean(density_rank4):.1f}")
    print(f"  Average non-zeros in lower-rank matrices: {np.mean(density_other):.1f}")
    
    # Distribution of number of each element
    print("\nElement distribution in rank-4 matrices:")
    zeros = [16 - d for d in density_rank4]
    print(f"  Average zeros: {np.mean(zeros):.1f}")
    print(f"  Std dev: {np.std(zeros):.1f}")

def propose_optimizations():
    """Propose computational optimizations for F_3 search."""
    print("\n\nPROPOSED OPTIMIZATIONS")
    print("="*50)
    
    print("\n1. SAMPLING APPROACH:")
    print("   - Don't check all matrices, sample intelligently")
    print("   - Focus on density regions likely to have counterexamples")
    print("   - Use importance sampling based on F_2 results")
    
    print("\n2. ALGEBRAIC OPTIMIZATIONS:")
    print("   - Use GL(4,3) orbits to reduce search space by ~300x")
    print("   - Exploit automorphisms of F_3")
    print("   - Use canonical forms")
    
    print("\n3. PARALLEL COMPUTATION:")
    print("   - Embarrassingly parallel over matrix subsets")
    print("   - GPU acceleration for rank computations")
    print("   - Distributed computing possible")
    
    print("\n4. EARLY TERMINATION:")
    print("   - If we find patterns like in F_2, stop early")
    print("   - Focus on boundary cases (dense matrices)")
    
    print("\n5. THEORETICAL INSIGHTS:")
    print("   - The zero-free obstruction likely persists")
    print("   - Matrices with few zeros probably counterexamples")
    print("   - Could prove results for zero-free case")

def main():
    print("FEASIBILITY ANALYSIS: HADAMARD FACTORIZATION OVER F_3")
    print("="*70)
    
    # Count matrices
    rank4_count, rank2_count = count_matrices_by_rank_f3()
    
    # Analyze cost
    pairs = analyze_computational_cost(rank4_count, rank2_count)
    
    # Test patterns
    test_density_patterns()
    
    # Propose optimizations
    propose_optimizations()
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("Full exhaustive search is computationally challenging but not impossible.")
    print("With optimizations and sampling, we could get meaningful results!")
    print("\nRecommended approach:")
    print("1. Start with theoretical analysis of zero-free case")
    print("2. Use sampling to find counterexamples")
    print("3. Focus on dense matrices (few zeros)")
    print("4. Look for patterns to guide further search")

if __name__ == "__main__":
    main()