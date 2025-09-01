# Speeding Up F_3 Search on Colab

## The Problem
The NumPy rank computation is the bottleneck - it's sequential and can't leverage the GPU.

## Solutions

### Option 1: Numba JIT Compilation (10-100x speedup)
**File**: `f3_numba_fast.py`

**Colab setup**:
```python
# Install Numba (usually pre-installed on Colab)
!pip install numba

# Install JAX with GPU
!pip install -q jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Upload and run
from google.colab import files
uploaded = files.upload()  # Upload f3_numba_fast.py
!python f3_numba_fast.py
```

**Benefits**:
- Numba compiles the rank function to machine code
- Parallel processing using `prange`
- Should run in 30-60 minutes instead of hours

### Option 2: Multi-core Parallel Processing
**File**: `f3_optimized_gpu.py`

**Benefits**:
- Uses Python multiprocessing for rank computation
- Distributes work across all CPU cores
- GPU still handles matrix operations

### Option 3: Cython (if Numba doesn't work)

Create a Cython version if needed:
```python
%%writefile rank_f3_cython.pyx
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def rank_f3_cython(long M_int):
    cdef int[4][4] M
    cdef int i, j, r, c, rank, row, col, pivot, inv, factor
    
    # Extract matrix
    for i in range(16):
        M[i // 4][i % 4] = (M_int // (3**i)) % 3
    
    # Gaussian elimination
    rank = 0
    for col in range(4):
        for row in range(rank, 4):
            if M[row][col] % 3 != 0:
                # Swap rows
                if row != rank:
                    for c in range(4):
                        M[rank][c], M[row][c] = M[row][c], M[rank][c]
                
                # Make pivot 1
                pivot = M[rank][col] % 3
                inv = 1 if pivot == 1 else 2
                for c in range(4):
                    M[rank][c] = (M[rank][c] * inv) % 3
                
                # Eliminate
                for r in range(4):
                    if r != rank and M[r][col] % 3 != 0:
                        factor = M[r][col] % 3
                        for c in range(4):
                            M[r][c] = (M[r][c] - factor * M[rank][c]) % 3
                
                rank += 1
                break
    
    return rank
```

Then compile:
```python
!pip install cython
from pyximport import pyxbuild
pyxbuild.install()
import rank_f3_cython
```

## Performance Expectations

| Method | Estimated Time | Speedup |
|--------|---------------|---------|
| Pure NumPy | 8-10 hours | 1x |
| Numba JIT | 30-60 min | 10-20x |
| Multiprocessing | 2-3 hours | 3-4x |
| Cython | 1-2 hours | 5-8x |

## Recommendations

1. **Try Numba first** (`f3_numba_fast.py`) - easiest and fastest
2. If Numba fails, use multiprocessing (`f3_optimized_gpu.py`)
3. As a last resort, implement Cython version

## Memory Management

If you run out of memory:
1. Reduce batch sizes in the code
2. Use `gc.collect()` periodically
3. Clear JAX caches: `jax.clear_caches()`
4. Save intermediate results and process in stages

## Monitoring Progress

The code shows:
- Progress bars with ETA
- Matrices/second processing rate
- Memory usage (if you add monitoring)

## Quick Test

Before running the full search, test with smaller ranges:
```python
# Modify the code to test first million matrices only
n_total = 10**6  # Instead of 3**16
```