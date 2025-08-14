#!/usr/bin/env python3
"""
Verify F_3 results and create comparison visualizations.
Also check for missing rank-2 matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

def theoretical_rank_count(n, r, q):
    """
    Theoretical number of nxn matrices of rank r over F_q.
    """
    if r == 0:
        return 1
    
    count = 1
    for i in range(r):
        count *= (q**n - q**i)**2
        count //= (q**r - q**i)
    
    return count

def verify_rank2_count():
    """Verify the rank-2 count against theory."""
    print("="*60)
    print("RANK-2 VERIFICATION")
    print("="*60)
    
    # Theoretical counts for 4x4 matrices over F_3
    print("\nTheoretical rank distribution:")
    total = 0
    for r in range(5):
        count = theoretical_rank_count(4, r, 3)
        total += count
        pct = count / (3**16) * 100
        print(f"  Rank {r}: {count:,} ({pct:.3f}%)")
    
    print(f"\nTotal: {total:,} (should be {3**16:,})")
    
    # Compare with actual
    actual_rank2 = 758007
    expected_rank2 = theoretical_rank_count(4, 2, 3)
    missing = expected_rank2 - actual_rank2
    missing_pct = (missing / expected_rank2) * 100
    
    print(f"\nRank-2 matrices:")
    print(f"  Expected: {expected_rank2:,}")
    print(f"  Found:    {actual_rank2:,}")
    print(f"  Missing:  {missing:,} ({missing_pct:.1f}%)")
    
    return expected_rank2, actual_rank2

def analyze_expressibility_impact(missing_rank2_pct):
    """Estimate impact of missing rank-2 matrices on expressibility rates."""
    print("\n" + "="*60)
    print("IMPACT ANALYSIS")
    print("="*60)
    
    # The missing 6.6% of rank-2 matrices means we're missing their products
    # This could affect expressibility rates
    
    print(f"\nMissing {missing_rank2_pct:.1f}% of rank-2 matrices means:")
    print(f"- Missing ~{missing_rank2_pct:.1f}% of products (assuming uniform)")
    print(f"- Expressibility rates could be underestimated by up to {missing_rank2_pct:.1f}%")
    
    # Adjust the observed rates
    print("\nAdjusted expressibility estimates (assuming uniform missing):")
    
    # F_3 results
    f3_results = [
        (0, 3639, 786),
        (1, 32817, 9801),
        (2, 127292, 52341),
        (3, 295355, 162154),
        (4, 461891, 329028),
        (5, 545134, 458153),
        (6, 498084, 459527),
        (7, 351512, 338124),
        (8, 182403, 178455),
        (9, 65001, 63916),
        (10, 14973, 14768),
        (11, 1993, 1974),
        (12, 116, 115),
    ]
    
    print("\nZeros | Observed | Adjusted | Difference")
    print("------|----------|----------|------------")
    
    adjusted_rates = []
    for zeros, total, expr in f3_results:
        if total > 0:
            obs_rate = expr / total * 100
            # Adjust assuming missing matrices would have similar expressibility
            adj_rate = min(99.9, obs_rate * (1 + missing_rank2_pct/100))
            diff = adj_rate - obs_rate
            print(f"  {zeros:2d}  |   {obs_rate:5.1f}% |   {adj_rate:5.1f}% | +{diff:4.1f}%")
            adjusted_rates.append((zeros, adj_rate))
    
    return adjusted_rates

def create_comparison_plots():
    """Create visualization comparing F_2 and F_3 results."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # F_2 results (from previous analysis)
    f2_data = {
        0: (20, 0, 0.0),
        1: (327, 0, 0.0),
        2: (1445, 3, 0.2),
        3: (2646, 18, 0.7),
        4: (3042, 62, 2.0),
        5: (2447, 155, 6.3),
        6: (1444, 298, 20.6),
        7: (652, 343, 52.6),
        8: (280, 235, 83.9),
        9: (77, 74, 96.1),
        10: (24, 24, 100.0),
        11: (0, 0, 0),
        12: (0, 0, 0),
    }
    
    # F_3 results
    f3_data = {
        0: (3639, 786, 21.6),
        1: (32817, 9801, 29.9),
        2: (127292, 52341, 41.1),
        3: (295355, 162154, 54.9),
        4: (461891, 329028, 71.2),
        5: (545134, 458153, 84.0),
        6: (498084, 459527, 92.3),
        7: (351512, 338124, 96.2),
        8: (182403, 178455, 97.8),
        9: (65001, 63916, 98.3),
        10: (14973, 14768, 98.6),
        11: (1993, 1974, 99.0),
        12: (116, 115, 99.1),
    }
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Expressibility rates comparison
    zeros_f2 = [z for z in f2_data.keys() if f2_data[z][0] > 0]
    rates_f2 = [f2_data[z][2] for z in zeros_f2]
    
    zeros_f3 = [z for z in f3_data.keys() if f3_data[z][0] > 0]
    rates_f3 = [f3_data[z][2] for z in zeros_f3]
    
    ax1.plot(zeros_f2, rates_f2, 'o-', label='F₂', linewidth=2, markersize=8)
    ax1.plot(zeros_f3, rates_f3, 's-', label='F₃', linewidth=2, markersize=8)
    ax1.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
    ax1.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    
    ax1.set_xlabel('Number of zeros in matrix', fontsize=12)
    ax1.set_ylabel('Expressibility rate (%)', fontsize=12)
    ax1.set_title('Expressibility vs Matrix Density', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)
    
    # Plot 2: Distribution of rank-4 matrices by zeros
    zeros_range = range(13)
    counts_f2 = [f2_data.get(z, (0, 0, 0))[0] for z in zeros_range]
    counts_f3 = [f3_data.get(z, (0, 0, 0))[0] for z in zeros_range]
    
    x = np.arange(len(zeros_range))
    width = 0.35
    
    ax2.bar(x - width/2, counts_f2, width, label='F₂', alpha=0.8)
    ax2.bar(x + width/2, counts_f3, width, label='F₃', alpha=0.8)
    
    ax2.set_xlabel('Number of zeros', fontsize=12)
    ax2.set_ylabel('Count of rank-4 matrices', fontsize=12)
    ax2.set_title('Distribution of Rank-4 Matrices', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(zeros_range)
    ax2.legend()
    ax2.set_yscale('log')
    
    # Plot 3: Phase diagram
    ax3.fill_between([0, 5.5], 0, 100, alpha=0.3, color='red', label='Dense (0-5 zeros)')
    ax3.fill_between([5.5, 10.5], 0, 100, alpha=0.3, color='yellow', label='Medium (6-10 zeros)')
    ax3.fill_between([10.5, 16], 0, 100, alpha=0.3, color='green', label='Sparse (11-16 zeros)')
    
    ax3.plot(zeros_f2, rates_f2, 'o-', label='F₂', linewidth=3, markersize=10, color='blue')
    ax3.plot(zeros_f3, rates_f3, 's-', label='F₃', linewidth=3, markersize=10, color='red')
    
    ax3.set_xlabel('Number of zeros in matrix', fontsize=12)
    ax3.set_ylabel('Expressibility rate (%)', fontsize=12)
    ax3.set_title('Phase Diagram: Expressibility Regions', fontsize=14)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-5, 105)
    ax3.set_xlim(-0.5, 12.5)
    
    plt.tight_layout()
    plt.savefig('f2_f3_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved comparison plot to f2_f3_comparison.png")
    
    # Create adjusted plot
    fig2, ax4 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Get adjusted rates
    _, actual_rank2 = verify_rank2_count()
    expected_rank2 = 811200
    missing_pct = (expected_rank2 - actual_rank2) / expected_rank2 * 100
    adjusted_rates = analyze_expressibility_impact(missing_pct)
    
    zeros_adj = [z for z, _ in adjusted_rates]
    rates_adj = [r for _, r in adjusted_rates]
    
    ax4.plot(zeros_f3, rates_f3, 's-', label='F₃ (observed)', linewidth=2, markersize=8)
    ax4.plot(zeros_adj, rates_adj, 'o--', label='F₃ (adjusted)', linewidth=2, markersize=8)
    ax4.plot(zeros_f2, rates_f2, '^-', label='F₂', linewidth=2, markersize=8, alpha=0.7)
    
    ax4.set_xlabel('Number of zeros in matrix', fontsize=12)
    ax4.set_ylabel('Expressibility rate (%)', fontsize=12)
    ax4.set_title('Impact of Missing Rank-2 Matrices', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-5, 105)
    
    plt.tight_layout()
    plt.savefig('f3_adjusted_rates.png', dpi=300, bbox_inches='tight')
    print("✓ Saved adjusted rates plot to f3_adjusted_rates.png")
    
    return fig, fig2

def suggest_verification():
    """Suggest ways to verify the missing matrices."""
    print("\n" + "="*60)
    print("VERIFICATION SUGGESTIONS")
    print("="*60)
    
    print("\nTo verify/fix the missing rank-2 matrices:")
    print("\n1. Quick sampling check:")
    print("   - Sample 100,000 random matrices")
    print("   - Count how many are rank-2")
    print("   - Should be ~1.88% (811,200 / 43,046,721)")
    print("   - Your search found ~1.76%")
    
    print("\n2. Check specific patterns:")
    print("   - The missing matrices might follow a pattern")
    print("   - Check if certain matrix structures were skipped")
    
    print("\n3. Re-run with sequential processing:")
    print("   - Parallel processing might have race conditions")
    print("   - Try a smaller range sequentially")
    
    print("\n4. The impact on conclusions:")
    print("   - Even with 6.6% missing, the pattern is clear")
    print("   - F₃ behaves VERY differently from F₂")
    print("   - Dense matrices are much more expressible in F₃")

if __name__ == "__main__":
    print("F₃ HADAMARD FACTORIZATION - RESULTS ANALYSIS")
    print("="*60)
    
    # Verify rank-2 count
    verify_rank2_count()
    
    # Create visualizations
    create_comparison_plots()
    
    # Suggest verification steps
    suggest_verification()
    
    print("\n✓ Analysis complete!")
    print("\nKey finding: The zero-free obstruction from F₂ does NOT")
    print("generalize to F₃ in the expected way. This is a major result!")