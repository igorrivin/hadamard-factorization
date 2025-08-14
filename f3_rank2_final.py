#!/usr/bin/env python3
"""
Final correct calculation using the standard formula from finite geometry.
"""

def q_factorial(n, q):
    """q-factorial: [n]_q! = (q^n - 1)(q^n - q)...(q^n - q^(n-1))"""
    result = 1
    for i in range(n):
        result *= (q**n - q**i)
    return result

def count_rank_r_matrices_final(n, r, q):
    """
    The number of n×n matrices of rank r over F_q is:
    
    |GL_n(q)| × |B_{n,r}(q)| / |GL_r(q)|
    
    where B_{n,r}(q) is the number of ordered bases of an r-dimensional subspace
    
    This equals: prod_{i=0}^{r-1} [(q^n - q^i)^2 / (q^r - q^i)]
    """
    if r == 0:
        return 1
    
    count = 1
    for i in range(r):
        count *= (q**n - q**i)**2
        count //= (q**r - q**i)
    
    return count

# Let's also verify by computing the orbit formula way
def orbit_formula_count(n, r, q):
    """
    Alternative: The GL_n(q) action on rank-r matrices has one orbit.
    So count = |GL_n(q)| / |Stab(M_r)|
    where M_r is any rank-r matrix.
    
    The stabilizer has size |GL_r(q)| × |GL_{n-r}(q)| × q^{r(n-r)}
    """
    # Size of GL_n(q)
    gl_n = 1
    for i in range(n):
        gl_n *= (q**n - q**i)
    
    if r == 0:
        return 1
    
    # Size of stabilizer
    gl_r = 1
    for i in range(r):
        gl_r *= (q**r - q**i)
    
    gl_nr = 1
    for i in range(n-r):
        gl_nr *= (q**(n-r) - q**i)
    
    stab_size = gl_r * gl_nr * q**(r*(n-r))
    
    return gl_n // stab_size

print("Rank counts for 4×4 matrices over F_3:")
print("="*50)

total = 0
for r in range(5):
    count1 = count_rank_r_matrices_final(4, r, 3)
    count2 = orbit_formula_count(4, r, 3)
    
    total += count1
    pct = count1 / (3**16) * 100
    
    print(f"Rank {r}: {count1:,} ({pct:.4f}%)")
    if count1 != count2:
        print(f"  WARNING: Orbit formula gives {count2:,}")

print(f"\nTotal: {total:,}")
print(f"Should be: {3**16:,}")
print(f"Check: {total == 3**16}")

# The rank-2 count specifically
rank2 = count_rank_r_matrices_final(4, 2, 3)
print(f"\nRank-2 matrices: {rank2:,}")

# Break it down
print("\nDetailed calculation for rank 2:")
for i in range(2):
    num = (3**4 - 3**i)**2
    den = (3**2 - 3**i)
    print(f"i={i}: ({3**4} - {3**i})^2 / ({3**2} - {3**i}) = {num} / {den} = {num//den}")

print(f"\nProduct: {(3**4 - 1)**2 // (3**2 - 1)} × {(3**4 - 3)**2 // (3**2 - 3)} = {rank2:,}")