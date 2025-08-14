#!/usr/bin/env python3
"""
Theoretical calculation of rank-2 matrices over F_3.
"""

def count_rank_k_matrices(n, k, q):
    """
    Count n×n matrices of rank k over F_q.
    
    Formula: |GL_n(F_q)| * C(n,k)_q * C(n,k)_q / |GL_k(F_q)|
    where C(n,k)_q is q-binomial coefficient
    """
    
    # GL_n(F_q) size
    gl_n = 1
    for i in range(n):
        gl_n *= (q**n - q**i)
    
    # GL_k(F_q) size  
    gl_k = 1
    for i in range(k):
        gl_k *= (q**k - q**i)
    
    # q-binomial coefficient [n choose k]_q
    qbinom = 1
    for i in range(k):
        qbinom *= (q**(n-i) - 1) // (q**(i+1) - 1)
    
    # Total count
    count = gl_n * qbinom * qbinom // gl_k
    
    return count

# For 4×4 matrices over F_3
n = 4
q = 3

print("Theoretical counts for 4×4 matrices over F_3:")
print("="*50)

total = q**(n*n)
print(f"Total matrices: {total:,}")

for k in range(5):
    if k == 0:
        count = 1  # Only zero matrix
    else:
        count = count_rank_k_matrices(n, k, q)
    
    percentage = count / total * 100
    print(f"Rank {k}: {count:,} ({percentage:.2f}%)")

# Alternative calculation for rank 2
print("\nAlternative calculation for rank 2:")

# Number of ways to choose 2 linearly independent rows
# First row: 3^4 - 1 choices (non-zero)
# Second row: 3^4 - 3 choices (not multiple of first)
row_choices = (3**4 - 1) * (3**4 - 3)

# Ways to express other 2 rows as linear combinations
# Each row: 3^2 = 9 choices of coefficients
lincomb_choices = 3**2 * 3**2

# But we've overcounted due to GL_2(F_3) action
gl2_size = (3**2 - 1) * (3**2 - 3) # = 8 * 6 = 48

rank2_count = row_choices * lincomb_choices // gl2_size
print(f"Rank 2 count: {rank2_count:,}")

# Sanity check with direct enumeration formula
def rank2_direct():
    """Direct calculation."""
    # Choose 2-dimensional row space
    # This is the Grassmannian Gr(2,4) over F_3
    
    # Size of Grassmannian
    gr_24 = 1
    for i in range(2):
        gr_24 *= (3**4 - 3**i) // (3**2 - 3**i)
    
    # Each 2-dimensional subspace gives 3^2 * 3^2 matrices
    # (choose which linear combination for each of 4 rows)
    return gr_24 * (3**2)**4

print(f"\nDirect Grassmannian calculation: {rank2_direct():,}")