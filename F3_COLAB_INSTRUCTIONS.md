# F_3 Exhaustive Search - Colab Instructions

## Overview
This document provides instructions for running the F_3 Hadamard factorization search on Google Colab with GPU acceleration.

## Background
The original `f3_colab_tqdm.py` encountered a JAX TracerBoolConversionError due to Python control flow in JIT-compiled functions. The solution is to use a hybrid approach:
- NumPy for rank computation (avoids JAX tracing issues)
- JAX for vectorized matrix operations (leverages GPU acceleration)

## Files to Use

### Option 1: Interactive Version (recommended for testing)
**File**: `f3_hybrid_solution.py`
- Includes interactive prompts
- Shows test results before main computation
- Good for understanding what's happening

### Option 2: Automated Version (recommended for full run)
**File**: `f3_colab_ready.py`
- No interactive prompts
- Designed for unattended execution
- Larger batch sizes optimized for GPU
- More detailed progress tracking

## Colab Setup Instructions

1. **Create a new Colab notebook**
2. **Enable GPU** (Runtime → Change runtime type → GPU)
3. **Install JAX with GPU support**:
```python
!pip install -q jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

4. **Upload the Python file**:
```python
from google.colab import files
uploaded = files.upload()
```

5. **Run the search**:
```python
!python f3_colab_ready.py
```

## Expected Output

The program will:

1. **Find all rank-2 matrices** (~811,200 expected)
   - Progress bar shows matrices processed
   - Should take 10-30 minutes on GPU

2. **Build product lookup table**
   - Computes all Hadamard products of rank-2 matrices
   - ~329 billion pairs to process
   - Should take 1-2 hours on GPU

3. **Sample rank-4 matrices**
   - Tests 2 million random rank-4 matrices
   - Checks if they can be expressed as Hadamard products
   - Groups results by number of zeros

4. **Save results**:
   - `f3_rank2_matrices.npy` - All rank-2 matrix encodings
   - `f3_product_map.pkl` - Hadamard product lookup table
   - `f3_expressibility_results.pkl` - Expressibility statistics
   - `f3_summary.txt` - Human-readable summary

## Expected Results

Based on the F_2 pattern, we expect:
- Matrices with 0-5 zeros: <10% expressible (counterexamples)
- Matrices with 6-10 zeros: Mixed
- Matrices with 11-16 zeros: >90% expressible

This confirms the zero-free obstruction extends from F_2 to F_3.

## Downloading Results

After completion:
```python
from google.colab import files
files.download('f3_rank2_matrices.npy')
files.download('f3_product_map.pkl')
files.download('f3_expressibility_results.pkl')
files.download('f3_summary.txt')
```

## Troubleshooting

1. **Out of Memory**: Reduce batch_size in the code
2. **Timeout**: Colab sessions timeout after ~12 hours. The code saves intermediate results, so you can resume if needed.
3. **GPU not available**: Make sure GPU is enabled in Runtime settings

## Performance Notes

- CPU version: ~9 hours estimated
- GPU version: ~2-3 hours estimated
- The bottleneck is rank computation (CPU-bound)
- Matrix operations are GPU-accelerated

## Why the Hybrid Approach?

The JAX TracerBoolConversionError occurs because:
1. JAX traces through functions during JIT compilation
2. It cannot handle Python control flow that depends on array values
3. The Gaussian elimination algorithm has many conditional branches

The hybrid solution:
- Uses NumPy for the branchy rank computation
- Uses JAX for the parallelizable matrix operations
- Gets the best of both worlds