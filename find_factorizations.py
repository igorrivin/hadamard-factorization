#!/usr/bin/env python3
"""
Find and display explicit factorizations for expressible matrices.
"""

import numpy as np
import pickle
from collections import defaultdict
import time

def rank_f2_fast(matrix_int):
    """Fast rank computation over F_2."""
    rows = []
    for i in range(4):
        rows.append((matrix_int >> (4*i)) & 0xF)
    
    rank = 0
    for col in range(4):
        col_mask = 1 << col
        for row_idx in range(rank, 4):
            if rows[row_idx] & col_mask:
                if row_idx != rank:
                    rows[rank], rows[row_idx] = rows[row_idx], rows[rank]
                for r in range(4):
                    if r != rank and rows[r] & col_mask:
                        rows[r] ^= rows[rank]
                rank += 1
                break
    return rank

def int_to_matrix(n):
    """Convert integer to 4x4 binary matrix."""
    M = np.zeros((4, 4), dtype=int)
    for i in range(16):
        M[i//4, i%4] = (n >> i) & 1
    return M

def find_all_factorizations(target, rank2_list, max_results=3):
    """Find all factorizations of target matrix."""
    factorizations = []
    
    # Group rank-2 matrices by their Hadamard product with each other
    for i, a in enumerate(rank2_list):
        if len(factorizations) >= max_results:
            break
            
        for j in range(i, len(rank2_list)):
            b = rank2_list[j]
            if (a & b) == target:
                factorizations.append((a, b))
                if len(factorizations) >= max_results:
                    break
    
    return factorizations

def main():
    print("FINDING HADAMARD FACTORIZATIONS")
    print("="*50)
    
    # Load results
    try:
        with open('expressible_complete.pkl', 'rb') as f:
            expressible = pickle.load(f)
        print(f"Loaded {len(expressible)} expressible matrices")
    except FileNotFoundError:
        print("Error: Run hadamard_search.py first!")
        return
    
    # Find all rank-2 matrices
    print("\nFinding all rank-2 matrices over F_2...")
    start = time.time()
    rank2_matrices = []
    for n in range(2**16):
        if rank_f2_fast(n) == 2:
            rank2_matrices.append(n)
    
    print(f"Found {len(rank2_matrices)} rank-2 matrices in {time.time()-start:.1f}s")
    
    # Create reverse lookup: for each expressible matrix, store its factorizations
    print("\nBuilding factorization database...")
    factorization_db = defaultdict(list)
    
    # This is the key insight: we can build this efficiently
    checked = 0
    for i, a in enumerate(rank2_matrices):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(rank2_matrices)}")
        
        for j in range(i, len(rank2_matrices)):
            b = rank2_matrices[j]
            product = a & b
            
            # Check if product has rank 4
            if rank_f2_fast(product) == 4:
                factorization_db[product].append((a, b))
            
            checked += 1
    
    print(f"Found factorizations for {len(factorization_db)} matrices")
    
    # Display examples
    print("\n" + "="*50)
    print("EXAMPLE FACTORIZATIONS")
    
    with open('detailed_factorizations.txt', 'w') as f:
        f.write("Detailed Hadamard Factorizations\n")
        f.write("="*60 + "\n\n")
        
        # Show some interesting examples
        examples = sorted(factorization_db.keys())[:10]
        
        for i, matrix_int in enumerate(examples):
            M = int_to_matrix(matrix_int)
            factorizations = factorization_db[matrix_int][:3]  # Show up to 3
            
            print(f"\nExample {i+1}: Matrix with {len(factorization_db[matrix_int])} factorizations")
            print(M)
            
            f.write(f"Matrix {i+1} (int={matrix_int}):\n")
            f.write(f"Total factorizations: {len(factorization_db[matrix_int])}\n")
            f.write(f"M =\n{M}\n\n")
            
            for j, (a, b) in enumerate(factorizations):
                A = int_to_matrix(a)
                B = int_to_matrix(b)
                
                f.write(f"Factorization {j+1}:\n")
                f.write(f"A (rank {rank_f2_fast(a)}) =\n{A}\n")
                f.write(f"B (rank {rank_f2_fast(b)}) =\n{B}\n")
                f.write(f"Verification: A ∘ B =\n{A * B}\n\n")
            
            f.write("-"*60 + "\n\n")
    
    # Statistics about factorizations
    print("\n" + "="*50)
    print("FACTORIZATION STATISTICS")
    
    factorization_counts = defaultdict(int)
    for m, facts in factorization_db.items():
        factorization_counts[len(facts)] += 1
    
    print("\nNumber of factorizations per matrix:")
    for count in sorted(factorization_counts.keys())[:10]:
        num_matrices = factorization_counts[count]
        print(f"  {count} factorizations: {num_matrices} matrices")
    
    # Find matrix with most factorizations
    max_facts = 0
    max_matrix = None
    for m, facts in factorization_db.items():
        if len(facts) > max_facts:
            max_facts = len(facts)
            max_matrix = m
    
    print(f"\nMatrix with most factorizations ({max_facts}):")
    print(int_to_matrix(max_matrix))
    
    print("\nSaved detailed examples to: detailed_factorizations.txt")

if __name__ == "__main__":
    main()