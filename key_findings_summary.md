# Key Findings Summary: Hadamard Factorization Research

## Major Discoveries

### 1. **Field Characteristic Matters Dramatically**

#### F₂ (Binary Field):
- **Zero-free obstruction**: 0% of all-1s matrices are factorizable
- **Density predicts factorizability**: 95.7% correlation
- Dense matrices (15-16 ones) are rarely expressible

#### F₃ (Ternary Field):
- **Complete reversal**: Zero-free matrices are 21% factorizable!
- Dense matrices show 21-84% expressibility
- The zero-free obstruction is specific to characteristic 2

### 2. **The U-Shaped Curve Over Reals**

Success rates by number of zeros:
```
0 zeros:  48% ✓ (surprisingly high!)
1 zero:   18%
2 zeros:   0%
3-8 zeros: 0-6% ← "Death Valley"
9 zeros:   3%
10 zeros:  8%
11 zeros: 28%
12 zeros: 50% ✓
```

### 3. **Pattern Type Trumps Zero Count**

Within each sparsity level:
- **Empty rows/columns**: 15-100% factorizable (huge improvement!)
- **Diagonal patterns**: Only slight improvement (up to 18%)
- **Block patterns**: Moderate improvement
- **Random scattered**: Usually not factorizable in death valley

### 4. **Specific Factorizable Patterns Found**

Even in the "death valley" (3-8 zeros), these patterns work:
- 4 zeros: Full row of zeros → 100% success
- 5 zeros: Specific column patterns → 80-100% success
- 8 zeros: Two empty rows/columns → 100% success

### 5. **Rigorous Certification Achieved**

- Implemented Smale's α-criterion
- 89% of factorized matrices are rigorously certified
- Handles floating-point errors and near-degeneracies

## Theoretical Implications

1. **Characteristic 2 is Special**: The zero-free obstruction appears unique to F₂
2. **Geometry of Factorizability**: The U-shape suggests interesting variety structure
3. **Constraint Analysis**: Empty rows/columns reduce the problem dimension favorably

## Practical Applications

1. **Interactive Tool**: Created Gradio app for testing any 4×4 matrix
2. **Pattern Recognition**: Can predict factorizability from zero pattern
3. **Efficient Algorithm**: Alternating projection with multiple initializations

## Open Questions

1. Why exactly does characteristic 2 behave differently?
2. What causes the U-shaped curve over the reals?
3. Can we find algebraic invariants that detect non-factorizability?
4. How does this generalize to larger matrices?

## Code Artifacts

- `hadamard_factorization_app.py`: Interactive Gradio application
- `analyze_sparsity_patterns.py`: Pattern-specific analysis
- `hadamard_rigorous_theory.py`: Certification framework
- `hadamard_writeup_augmented.tex`: Full mathematical write-up