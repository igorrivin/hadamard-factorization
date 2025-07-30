#!/usr/bin/env python3
"""
Exhaustive search for Hadamard product factorization over F_2.

This script determines which 4×4 full-rank binary matrices can be
expressed as Hadamard products of two rank-2 matrices.
"""

import numpy as np
import time
from collections import defaultdict

class HadamardSearchF2:
    def __init__(self):
        self.total_matrices = 2**16
        
    def rank_f2(self, matrix_int):
        """Compute rank of 4×4 matrix over F_2 using bit operations."""
        rows = []
        for i in range(4):
            rows.append((matrix_int >> (4*i)) & 0xF)
        
        rank = 0
        for col in range(4):
            col_mask = 1 << col
            # Find pivot
            found = False
            for row_idx in range(rank, 4):
                if rows[row_idx] & col_mask:
                    if row_idx != rank:
                        rows[rank], rows[row_idx] = rows[row_idx], rows[rank]
                    # Eliminate
                    for r in range(4):
                        if r != rank and rows[r] & col_mask:
                            rows[r] ^= rows[rank]
                    rank += 1
                    found = True
                    break
        return rank
    
    def int_to_matrix(self, n):
        """Convert 16-bit integer to 4×4 binary matrix."""
        M = np.zeros((4, 4), dtype=int)
        for i in range(16):
            M[i//4, i%4] = (n >> i) & 1
        return M
    
    def classify_matrices(self):
        """Classify all 2^16 matrices by rank."""
        print("Classifying all 65,536 matrices by rank...")
        rank_dict = defaultdict(list)
        
        for n in range(self.total_matrices):
            if n % 10000 == 0:
                print(f"Progress: {n}/{self.total_matrices}")
            r = self.rank_f2(n)
            rank_dict[r].append(n)
        
        print("\nClassification complete:")
        for r in range(5):
            print(f"Rank {r}: {len(rank_dict[r])} matrices")
        
        return rank_dict
    
    def find_hadamard_products(self, rank2_list, rank4_list):
        """Find which rank-4 matrices can be expressed as Hadamard products."""
        print("\nComputing Hadamard products of rank-2 matrices...")
        expressible = set()
        
        total_pairs = len(rank2_list) * (len(rank2_list) + 1) // 2
        checked = 0
        
        for i, a in enumerate(rank2_list):
            if i % 500 == 0:
                print(f"Progress: {i}/{len(rank2_list)} matrices ({checked}/{total_pairs} pairs)")
            
            for j in range(i, len(rank2_list)):
                b = rank2_list[j]
                # Hadamard product in F_2 is bitwise AND
                product = a & b
                
                # Check if product has rank 4
                if product in rank4_list:
                    expressible.add(product)
                
                checked += 1
        
        return expressible
    
    def search(self):
        """Main search algorithm."""
        start_time = time.time()
        
        # Step 1: Classify matrices
        rank_dict = self.classify_matrices()
        rank2_set = set(rank_dict[2])
        rank4_set = set(rank_dict[4])
        
        # Step 2: Find expressible matrices
        expressible = self.find_hadamard_products(rank_dict[2], rank4_set)
        
        # Step 3: Identify counterexamples
        counterexamples = rank4_set - expressible
        
        elapsed = time.time() - start_time
        
        # Report results
        print(f"\nSearch completed in {elapsed:.1f} seconds")
        print("\nFINAL RESULTS:")
        print(f"Total rank-4 matrices: {len(rank4_set)}")
        print(f"Expressible as rank-2 ∘ rank-2: {len(expressible)} ({len(expressible)/len(rank4_set)*100:.1f}%)")
        print(f"COUNTEREXAMPLES: {len(counterexamples)} ({len(counterexamples)/len(rank4_set)*100:.1f}%)")
        
        # Save counterexamples
        with open('counterexamples_f2.txt', 'w') as f:
            f.write(f"Found {len(counterexamples)} counterexamples over F_2\n\n")
            f.write("First 10 counterexamples:\n")
            for i, ce in enumerate(list(counterexamples)[:10]):
                M = self.int_to_matrix(ce)
                f.write(f"\nCounterexample {i+1} (int={ce}):\n")
                f.write(str(M) + "\n")
        
        # Save expressible matrices (non-counterexamples)
        with open('expressible_f2.txt', 'w') as f:
            f.write(f"Found {len(expressible)} expressible full-rank matrices over F_2\n\n")
            f.write("First 100 expressible matrices:\n")
            for i, exp in enumerate(list(expressible)[:100]):
                M = self.int_to_matrix(exp)
                f.write(f"\nExpressible matrix {i+1} (int={exp}):\n")
                f.write(str(M) + "\n")
        
        # Also save complete lists as binary files for further analysis
        import pickle
        
        with open('counterexamples_complete.pkl', 'wb') as f:
            pickle.dump(sorted(list(counterexamples)), f)
        
        with open('expressible_complete.pkl', 'wb') as f:
            pickle.dump(sorted(list(expressible)), f)
        
        # Show first counterexample
        if counterexamples:
            ce = list(counterexamples)[0]
            print(f"\nFirst counterexample (int={ce}):")
            print(self.int_to_matrix(ce))
        
        return len(counterexamples) == 5304  # Verify expected result

if __name__ == "__main__":
    print("HADAMARD PRODUCT FACTORIZATION OVER F_2")
    print("="*50)
    
    searcher = HadamardSearchF2()
    success = searcher.search()
    
    if success:
        print("\n✓ Result verified: Found exactly 5,304 counterexamples as expected!")
    else:
        print("\n⚠ Warning: Unexpected number of counterexamples!")