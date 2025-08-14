#!/usr/bin/env python3
"""
Fix the rank counting formula.
The correct formula for the number of m×n matrices of rank r over F_q is:

N(m,n,r,q) = q^(r(m+n-r)) * prod_{i=0}^{r-1} (q^m - q^i)(q^n - q^i) / (q^r - q^i)^2
"""

def count_matrices_rank_r(m, n, r, q):
    """
    Count m×n matrices of rank r over F_q.
    
    This uses the formula based on choosing r linearly independent rows,
    then r linearly independent columns, then filling in the r×r submatrix.
    """
    if r == 0:
        return 1  # Only zero matrix
    
    if r > min(m, n):
        return 0  # Impossible
    
    # The correct formula involves Gaussian polynomials
    # For square matrices, we use a different approach
    
    if m == n:
        # For n×n matrices of rank r
        # Formula: q^(r(n-r)) * prod_{i=0}^{r-1} [(q^n - q^i)^2 / (q^r - q^i)]
        
        result = q**(r * (n - r))
        
        for i in range(r):
            result *= (q**n - q**i)**2
            result //= (q**r - q**i)
        
        return result
    else:
        # General case
        result = q**(r * (m + n - r))
        
        for i in range(r):
            result *= (q**m - q**i) * (q**n - q**i)
            result //= (q**r - q**i)**2
        
        return result

# Verify with 2×2 
print("2×2 matrices over F_3:")
total_2x2 = 0
for r in range(3):
    count = count_matrices_rank_r(2, 2, r, 3)
    total_2x2 += count
    print(f"Rank {r}: {count}")
print(f"Total: {total_2x2} (should be {3**4})")

print("\n4×4 matrices over F_3:")
total_4x4 = 0
for r in range(5):
    count = count_matrices_rank_r(4, 4, r, 3)
    total_4x4 += count
    percentage = count / (3**16) * 100
    print(f"Rank {r}: {count:,} ({percentage:.4f}%)")
print(f"Total: {total_4x4:,} (should be {3**16:,})")

# Let's compute rank 2 step by step
print("\n" + "="*50)
print("Detailed rank-2 calculation:")

n = 4
r = 2
q = 3

# q^(r(n-r)) = 3^(2*2) = 3^4 = 81
base_term = q**(r * (n - r))
print(f"Base term: {q}^({r}×{n-r}) = {base_term}")

# Product for i=0: (q^n - q^0)^2 / (q^r - q^0) = (3^4 - 1)^2 / (3^2 - 1)
term0_num = (q**n - 1)**2
term0_den = (q**r - 1)
print(f"i=0: ({q**n} - 1)^2 / ({q**r} - 1) = {term0_num} / {term0_den} = {term0_num // term0_den}")

# Product for i=1: (q^n - q^1)^2 / (q^r - q^1) = (3^4 - 3)^2 / (3^2 - 3)
term1_num = (q**n - q)**2
term1_den = (q**r - q)
print(f"i=1: ({q**n} - {q})^2 / ({q**r} - {q}) = {term1_num} / {term1_den} = {term1_num // term1_den}")

# Total
rank2_count = base_term * (term0_num // term0_den) * (term1_num // term1_den)
print(f"\nTotal rank-2 matrices: {base_term} × {term0_num // term0_den} × {term1_num // term1_den} = {rank2_count:,}")

# Actually, I think there might be an issue with the formula. Let me try the standard one
print("\n" + "="*50)
print("Using standard Gaussian polynomial formula:")

def gaussian_poly(n, k, q):
    """Gaussian polynomial [n choose k]_q"""
    if k == 0 or k == n:
        return 1
    
    num = 1
    den = 1
    for i in range(k):
        num *= (q**(n-i) - 1)
        den *= (q**(i+1) - 1)
    
    return num // den

# Number of rank-r n×n matrices = [n choose r]_q^2 * q^(r^2) * prod_{i=1}^r (q^(n-r) - 1)
def rank_r_matrices_gaussian(n, r, q):
    if r == 0:
        return 1
    
    # Gaussian binomial squared
    gauss_sq = gaussian_poly(n, r, q)**2
    
    # q^(r^2)
    q_factor = q**(r**2)
    
    # Product term - actually this should be different
    # Let me use the most standard formula:
    # N(n,r,q) = prod_{i=0}^{r-1} (q^n - q^i)^2 / prod_{i=0}^{r-1} (q^r - q^i)
    
    num = 1
    den = 1
    for i in range(r):
        num *= (q**n - q**i)**2
        den *= (q**r - q**i)
    
    return num // den

print(f"\nUsing corrected formula for rank-2 4×4 over F_3:")
rank2_correct = rank_r_matrices_gaussian(4, 2, 3)
print(f"Rank-2 count: {rank2_correct:,}")