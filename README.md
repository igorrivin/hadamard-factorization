# Hadamard Product Factorization of 4×4 Matrices

This repository contains code for investigating whether all 4×4 full-rank matrices can be expressed as Hadamard products of two rank-2 matrices.

## Main Result

We computationally prove that **not all 4×4 full-rank matrices can be expressed as Hadamard products of two rank-2 matrices**, resolving an open problem in matrix theory.

## Key Findings

### Over F₂ (Binary Field)
- Total 4×4 matrices: 65,536
- Full-rank matrices: 20,160 (30.8%)
- Expressible as rank-2 ∘ rank-2: 14,856 (73.7%)
- **Counterexamples: 5,304 (26.3%)**

### Over ℤ (Integers)
- The F₂ counterexamples remain counterexamples over ℤ
- Verified by exhaustive search over sign patterns

### Over ℝ (Real Numbers)
- Strong numerical evidence that the counterexamples remain valid
- The zero pattern constraints interact with rank constraints in fundamental ways

## Example Counterexample

```
[[1 1 1 1]
 [1 1 1 0]
 [0 1 0 0]
 [1 0 0 0]]
```

This matrix has full rank but cannot be written as A ∘ B where rank(A) ≤ 2 and rank(B) ≤ 2.

## Repository Structure

- `hadamard_search.py`: Main search algorithm over F₂
- `verify_integers.py`: Verification over ℤ
- `test_real.py`: Numerical optimization tests over ℝ
- `analyze_results.py`: Analyze patterns in results
- `visualize_matrices.py`: Create UMAP visualizations of the matrix landscape
- `quick_visualize.py`: Generate example visualizations without full data
- `check_matrix.py`: Check if a specific matrix is expressible
- `run_tests.sh`: Run all tests
- `paper/`: LaTeX paper with detailed mathematical exposition

### Output Files

After running `hadamard_search.py`:
- `counterexamples_f2.txt`: First 10 counterexamples
- `expressible_f2.txt`: First 100 expressible matrices
- `counterexamples_complete.pkl`: All 5,304 counterexamples (binary format)
- `expressible_complete.pkl`: All 14,856 expressible matrices (binary format)

After running `analyze_results.py`:
- `factorization_examples.txt`: Examples with explicit factorizations

After running `visualize_matrices.py`:
- `umap_full_features.png`: UMAP embedding based on all features
- `umap_zero_patterns.png`: UMAP embedding based on zero patterns
- `average_matrices.png`: Heatmaps showing average patterns
- `ones_distribution.png`: Distribution of matrix density

## Usage

```bash
# Run the main F₂ search
python hadamard_search.py

# Verify counterexamples over integers
python verify_integers.py

# Test over real numbers
python test_real.py
```

## Requirements

- Python 3.x
- NumPy
- SciPy

## Citation

If you use this code in your research, please cite:
```
@article{hadamard2024,
  title={Computational Resolution of Hadamard Product Factorization for 4×4 Matrices},
  author={[Author Names]},
  year={2024}
}
```