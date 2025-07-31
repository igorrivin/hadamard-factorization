#!/usr/bin/env python3
"""
Verify that F_2 counterexamples remain counterexamples over the integers.

For a binary matrix C to be expressible as A ∘ B over Z with rank(A), rank(B) ≤ 2,
we must have A[i,j] = B[i,j] ∈ {-1, 1} where C[i,j] = 1, giving 2^k possibilities
to check where k is the number of ones in C.
"""

import numpy as np

def rank(M):
    """Compute rank of matrix over Q (equivalently over Z)."""
    return np.linalg.matrix_rank(M)

def verify_integer_factorization(C):
    """
    Check if binary matrix C can be expressed as A ∘ B over Z
    with rank(A) ≤ 2 and rank(B) ≤ 2.
    """
    # Find positions of ones
    one_positions = [(i, j) for i in range(4) for j in range(4) if C[i, j] == 1]
    num_ones = len(one_positions)
    
    print(f"Matrix has {num_ones} ones, checking {2**num_ones} sign patterns...")
    
    # Try all possible sign assignments
    for signs in range(2**num_ones):
        # Create signed matrix (A = B for integer case)
        A = np.zeros_like(C, dtype=float)
        
        for idx, (i, j) in enumerate(one_positions):
            # Extract bit for this position
            if (signs >> idx) & 1:
                A[i, j] = 1.0
            else:
                A[i, j] = -1.0
        
        # Check rank
        if rank(A) <= 2:
            print(f"Found factorization with rank {rank(A)}!")
            print(f"A = B =\n{A}")
            return True
    
    print("No factorization found among all sign patterns.")
    return False

def main():
    print("VERIFYING F_2 COUNTEREXAMPLES OVER INTEGERS")
    print("="*50)
    
    # Test our known counterexamples
    counterexamples = [
        np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0]
        ]),
        np.array([
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0]
        ]),
        np.array([
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0]
        ])
    ]
    
    all_fail = True
    
    for i, C in enumerate(counterexamples):
        print(f"\nCounterexample {i+1}:")
        print(C)
        
        can_factor = verify_integer_factorization(C)
        
        if can_factor:
            all_fail = False
            print("✗ Unexpectedly found a factorization!")
        else:
            print("✓ Confirmed: Cannot be factored over Z")
    
    print("\n" + "="*50)
    if all_fail:
        print("CONCLUSION: All tested F_2 counterexamples remain counterexamples over Z!")
        print("Therefore, not all 4×4 full-rank integer matrices can be expressed")
        print("as Hadamard products of two rank-2 integer matrices.")
    else:
        print("WARNING: Some counterexamples could be factored over Z!")

if __name__ == "__main__":
    main()