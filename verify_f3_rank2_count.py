#!/usr/bin/env python3
"""
Verify the rank-2 count for F_3 matrices.
"""

import numpy as np

def exact_rank_f3(M):
    """Exact rank computation over F_3."""
    M = M.copy() % 3
    rows, cols = M.shape
    rank = 0
    
    for col in range(cols):
        # Find pivot
        pivot_found = False
        for row in range(rank, rows):
            if M[row, col] % 3 != 0:
                # Swap rows
                M[[rank, row]] = M[[row, rank]]
                # Make pivot 1
                inv = 1 if M[rank, col] == 1 else 2
                M[rank] = (M[rank] * inv) % 3
                # Eliminate
                for r in range(rows):
                    if r != rank and M[r, col] != 0:
                        M[r] = (M[r] - M[r, col] * M[rank]) % 3
                rank += 1
                pivot_found = True
                break
    return rank

# Test small matrices first
print("Testing 2x2 matrices over F_3:")
rank_counts = {0: 0, 1: 0, 2: 0}
for a in range(3):
    for b in range(3):
        for c in range(3):
            for d in range(3):
                M = np.array([[a, b], [c, d]])
                r = exact_rank_f3(M)
                rank_counts[r] += 1

total = sum(rank_counts.values())
print(f"Total 2x2 matrices: {total}")
for r, count in rank_counts.items():
    print(f"  Rank {r}: {count} ({count/total*100:.1f}%)")

# Now estimate for 4x4
print("\n\nEstimating 4x4 rank-2 matrices:")

# Sample to estimate
n_samples = 100000
rank_counts_4x4 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

for _ in range(n_samples):
    M = np.random.randint(0, 3, (4, 4))
    r = exact_rank_f3(M)
    rank_counts_4x4[r] += 1

print(f"\nSampled {n_samples} random 4x4 matrices:")
for r, count in rank_counts_4x4.items():
    est_total = int(3**16 * count / n_samples)
    print(f"  Rank {r}: ~{est_total:,} ({count/n_samples*100:.2f}%)")

# The rank-2 probability for 4x4 over F_3
rank2_prob = rank_counts_4x4[2] / n_samples
est_rank2_total = int(3**16 * rank2_prob)
print(f"\nEstimated rank-2 4x4 matrices: {est_rank2_total:,}")

# Double-check with direct count of small sample
print("\nDirect count verification (first 10000 matrices):")
rank2_count = 0
for i in range(10000):
    # Convert i to base-3 representation
    digits = []
    n = i
    for _ in range(16):
        digits.append(n % 3)
        n //= 3
    M = np.array(digits).reshape(4, 4)
    
    if exact_rank_f3(M) == 2:
        rank2_count += 1

print(f"Found {rank2_count} rank-2 matrices in first 10000")
print(f"Extrapolated total: {int(rank2_count * 3**16 / 10000):,}")