#!/usr/bin/env python3
"""
Analyze the results of the Hadamard factorization search.
Load and examine both counterexamples and expressible matrices.
"""

import numpy as np
import pickle
from collections import defaultdict

def int_to_matrix(n):
    """Convert 16-bit integer to 4×4 binary matrix."""
    M = np.zeros((4, 4), dtype=int)
    for i in range(16):
        M[i//4, i%4] = (n >> i) & 1
    return M

def matrix_to_int(M):
    """Convert 4×4 binary matrix to 16-bit integer."""
    n = 0
    for i in range(4):
        for j in range(4):
            if M[i, j]:
                n |= (1 << (4*i + j))
    return n

def analyze_zero_patterns(matrix_list):
    """Analyze the distribution of zero patterns."""
    zero_counts = defaultdict(int)
    
    for n in matrix_list:
        M = int_to_matrix(n)
        num_zeros = np.sum(M == 0)
        zero_counts[num_zeros] += 1
    
    return dict(zero_counts)

def find_factorization(target_int, rank2_matrices):
    """Find a factorization if it exists."""
    target = int_to_matrix(target_int)
    
    for a in rank2_matrices:
        for b in rank2_matrices:
            if (a & b) == target_int:  # Hadamard product in F2
                return a, b
    return None, None

def main():
    print("ANALYZING HADAMARD FACTORIZATION RESULTS")
    print("="*50)
    
    try:
        # Load the complete lists
        with open('counterexamples_complete.pkl', 'rb') as f:
            counterexamples = pickle.load(f)
        
        with open('expressible_complete.pkl', 'rb') as f:
            expressible = pickle.load(f)
        
        print(f"\nLoaded {len(counterexamples)} counterexamples")
        print(f"Loaded {len(expressible)} expressible matrices")
        
    except FileNotFoundError:
        print("\nError: Run hadamard_search.py first to generate the result files!")
        return
    
    # Analyze zero patterns
    print("\nZero pattern distribution for COUNTEREXAMPLES:")
    ce_zeros = analyze_zero_patterns(counterexamples)
    for num_zeros in sorted(ce_zeros.keys()):
        count = ce_zeros[num_zeros]
        print(f"  {num_zeros} zeros: {count} matrices ({count/len(counterexamples)*100:.1f}%)")
    
    print("\nZero pattern distribution for EXPRESSIBLE matrices:")
    exp_zeros = analyze_zero_patterns(expressible)
    for num_zeros in sorted(exp_zeros.keys()):
        count = exp_zeros[num_zeros]
        print(f"  {num_zeros} zeros: {count} matrices ({count/len(expressible)*100:.1f}%)")
    
    # Compare average number of zeros
    ce_avg_zeros = sum(int_to_matrix(n).size - np.sum(int_to_matrix(n)) 
                      for n in counterexamples) / len(counterexamples)
    exp_avg_zeros = sum(int_to_matrix(n).size - np.sum(int_to_matrix(n)) 
                       for n in expressible) / len(expressible)
    
    print(f"\nAverage number of zeros:")
    print(f"  Counterexamples: {ce_avg_zeros:.2f}")
    print(f"  Expressible: {exp_avg_zeros:.2f}")
    
    # Show some specific examples
    print("\n" + "="*50)
    print("SPECIFIC EXAMPLES")
    
    # Find expressible matrix with most zeros
    max_zeros = -1
    max_zeros_matrix = None
    for n in expressible:
        M = int_to_matrix(n)
        num_zeros = np.sum(M == 0)
        if num_zeros > max_zeros:
            max_zeros = num_zeros
            max_zeros_matrix = n
    
    print(f"\nExpressible matrix with most zeros ({max_zeros} zeros):")
    print(int_to_matrix(max_zeros_matrix))
    
    # Find counterexample with fewest zeros
    min_zeros = 16
    min_zeros_matrix = None
    for n in counterexamples:
        M = int_to_matrix(n)
        num_zeros = np.sum(M == 0)
        if num_zeros < min_zeros:
            min_zeros = num_zeros
            min_zeros_matrix = n
    
    print(f"\nCounterexample with fewest zeros ({min_zeros} zeros):")
    print(int_to_matrix(min_zeros_matrix))
    
    # Save a sample of expressible matrices with their factorizations
    print("\n" + "="*50)
    print("SAVING FACTORIZATION EXAMPLES")
    
    # First, we need to find actual rank-2 matrices over F_2
    print("  Finding rank-2 matrices over F_2...")
    rank2_matrices = []
    
    def rank_f2(matrix_int):
        """Compute rank over F_2 using bit operations."""
        rows = []
        for i in range(4):
            rows.append((matrix_int >> (4*i)) & 0xF)
        
        rank = 0
        for col in range(4):
            col_mask = 1 << col
            found = False
            for row_idx in range(rank, 4):
                if rows[row_idx] & col_mask:
                    if row_idx != rank:
                        rows[rank], rows[row_idx] = rows[row_idx], rows[rank]
                    for r in range(4):
                        if r != rank and rows[r] & col_mask:
                            rows[r] ^= rows[rank]
                    rank += 1
                    found = True
                    break
        return rank
    
    # Find ALL rank-2 matrices (there are 7,350 of them)
    for n in range(2**16):
        if rank_f2(n) == 2:
            rank2_matrices.append(n)
    
    print(f"  Found {len(rank2_matrices)} rank-2 matrices")
    
    with open('factorization_examples.txt', 'w') as f:
        f.write("Examples of expressible matrices with their factorizations\n")
        f.write("="*60 + "\n\n")
        
        examples_found = 0
        # Convert rank2_matrices to a set for faster lookup
        rank2_set = set(rank2_matrices)
        
        print(f"  Searching for factorizations...")
        # Try some specific expressible matrices
        for i, exp in enumerate(sorted(expressible)[:20]):  # Check first 20
            if i % 5 == 0:
                print(f"    Checking matrix {i+1}...")
            
            found = False
            # Try factorizations more efficiently
            for a_idx, a in enumerate(rank2_matrices[:1000]):  # Limit search
                if found:
                    break
                for b in rank2_matrices[a_idx:a_idx+1000]:  # Check nearby matrices
                    if (a & b) == exp:  # Hadamard product in F2 is AND
                        M = int_to_matrix(exp)
                        A = int_to_matrix(a)
                        B = int_to_matrix(b)
                        
                        f.write(f"Matrix {examples_found+1} (int={exp}):\n")
                        f.write(f"M =\n{M}\n")
                        f.write(f"\nFactorization: M = A ∘ B where\n")
                        f.write(f"A (rank {rank_f2(a)}) =\n{A}\n")
                        f.write(f"B (rank {rank_f2(b)}) =\n{B}\n")
                        f.write(f"Verification: A ∘ B =\n{A * B}\n")
                        f.write("-"*40 + "\n\n")
                        
                        examples_found += 1
                        found = True
                        break
            
            if examples_found >= 5:
                break
    
    print(f"Saved {examples_found} factorization examples to factorization_examples.txt")

if __name__ == "__main__":
    main()