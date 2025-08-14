#!/usr/bin/env python3
"""
Correct calculation of rank-2 matrices over F_3.
Using the standard formula for matrices of given rank.
"""

def q_pochhammer(q, n, k):
    """(q^n)_k = (1 - q^n)(1 - q^(n-1))...(1 - q^(n-k+1))"""
    result = 1
    for i in range(k):
        result *= (q**(n-i) - 1)
    return result

def count_rank_k_matrices_correct(n, k, q):
    """
    Count n×n matrices of rank exactly k over F_q.
    
    The formula is:
    N(n,k,q) = q^(k(n-k)) * prod_{i=0}^{k-1} (q^n - q^i) / prod_{i=0}^{k-1} (q^k - q^i)
    
    Or equivalently:
    N(n,k,q) = q^(k(n-k)) * (q^n - 1)(q^n - q)...(q^n - q^(k-1)) / ((q^k - 1)(q^k - q)...(q^k - q^(k-1)))
    """
    if k == 0:
        return 1  # Only the zero matrix
    
    # Main term
    result = q**(k * (n - k))
    
    # Numerator: (q^n - 1)(q^n - q)...(q^n - q^(k-1))
    for i in range(k):
        result *= (q**n - q**i)
    
    # Denominator: (q^k - 1)(q^k - q)...(q^k - q^(k-1))
    for i in range(k):
        result //= (q**k - q**i)
    
    return result

# Test with known values first
print("Testing with 2×2 matrices over F_3:")
for k in range(3):
    count = count_rank_k_matrices_correct(2, k, 3)
    print(f"Rank {k}: {count}")
print(f"Total: {sum(count_rank_k_matrices_correct(2, k, 3) for k in range(3))}")
print(f"Should be: {3**4} = {3**4}")

print("\n" + "="*50)
print("4×4 matrices over F_3:")
print("="*50)

total = 0
for k in range(5):
    count = count_rank_k_matrices_correct(4, k, 3)
    percentage = count / (3**16) * 100
    print(f"Rank {k}: {count:,} ({percentage:.4f}%)")
    total += count

print(f"\nTotal: {total:,}")
print(f"Should be: 3^16 = {3**16:,}")
print(f"Difference: {3**16 - total}")

# Alternative verification using the explicit formula
print("\n" + "="*50)
print("Verification of rank-2 count:")

# For rank 2: q^(2*2) * [(q^4-1)(q^4-q)] / [(q^2-1)(q^2-q)]
q = 3
rank2_alt = (q**(2*2) * (q**4 - 1) * (q**4 - q)) // ((q**2 - 1) * (q**2 - q))
print(f"Rank 2 (alternative): {rank2_alt:,}")

# More explicit
numerator = 3**4 * (3**4 - 1) * (3**4 - 3)
denominator = (3**2 - 1) * (3**2 - 3)
print(f"\nDetailed calculation:")
print(f"3^4 = {3**4}")
print(f"3^4 - 1 = {3**4 - 1}")
print(f"3^4 - 3 = {3**4 - 3}")
print(f"3^2 - 1 = {3**2 - 1}")
print(f"3^2 - 3 = {3**2 - 3}")
print(f"Numerator: {3**4} × {3**4 - 1} × {3**4 - 3} = {numerator}")
print(f"Denominator: {3**2 - 1} × {3**2 - 3} = {denominator}")
print(f"Result: {numerator} / {denominator} = {numerator // denominator}")