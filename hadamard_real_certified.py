#!/usr/bin/env python3
"""
Certified Hadamard factorization solver with robust rank-2 parameterization.
Handles degenerate cases where standard basis rows are dependent.
"""

import numpy as np
from scipy.linalg import svd, qr
from itertools import combinations

def find_rank2_basis(M, tol=1e-10):
    """
    Find a robust basis for a rank-2 matrix.
    Returns (row_indices, basis_vectors, coefficients) such that
    all rows can be expressed as linear combinations of the basis.
    """
    n = M.shape[0]
    
    # Try all pairs of rows to find a linearly independent pair
    best_cond = float('inf')
    best_pair = None
    best_basis = None
    
    for i, j in combinations(range(n), 2):
        basis = np.array([M[i], M[j]])
        
        # Check linear independence using condition number
        G = basis @ basis.T  # Gram matrix
        try:
            cond = np.linalg.cond(G)
            if cond < best_cond and cond < 1/tol:
                best_cond = cond
                best_pair = (i, j)
                best_basis = basis
        except:
            continue
    
    if best_pair is None:
        # Matrix is rank-1 or rank-0
        return None, None, None
    
    # Express all rows in terms of the chosen basis
    i, j = best_pair
    basis = best_basis
    coeffs = np.zeros((n, 2))
    
    # The basis rows have trivial coefficients
    coeffs[i] = [1, 0]
    coeffs[j] = [0, 1]
    
    # Solve for other rows: row_k = c1*basis[0] + c2*basis[1]
    for k in range(n):
        if k not in (i, j):
            # Least squares solution
            c, _, _, _ = np.linalg.lstsq(basis.T, M[k], rcond=None)
            coeffs[k] = c
    
    return best_pair, basis, coeffs

def parameterize_rank2_robust(M):
    """
    Robustly parameterize a rank-2 matrix.
    Returns a dictionary with basis info and coefficients.
    """
    row_indices, basis, coeffs = find_rank2_basis(M)
    
    if row_indices is None:
        # Degenerate case - use SVD
        U, s, Vt = svd(M)
        rank = np.sum(s > 1e-10)
        
        if rank < 2:
            # Perturb to make it rank-2
            s[0] = max(s[0], 1e-6)
            s[1] = max(s[1], 1e-6)
            M = U[:, :2] @ np.diag(s[:2]) @ Vt[:2, :]
            return parameterize_rank2_robust(M)
    
    return {
        'row_indices': row_indices,
        'basis': basis,
        'coefficients': coeffs,
        'condition': np.linalg.cond(basis @ basis.T)
    }

def construct_from_params(basis_rows, coeffs):
    """Reconstruct matrix from basis and coefficients."""
    n = len(coeffs)
    M = np.zeros((n, basis_rows.shape[1]))
    
    for i in range(n):
        M[i] = coeffs[i, 0] * basis_rows[0] + coeffs[i, 1] * basis_rows[1]
    
    return M

def solve_hadamard_certified(C, max_iter=1000, tol=1e-10, verbose=True):
    """
    Certified Hadamard factorization with robust handling of degenerate cases.
    """
    n = C.shape[0]
    
    # Initialize with rank-2 approximation
    U, s, Vt = svd(C)
    s[2:] = 0
    A = U @ np.diag(s) @ Vt + 0.01 * np.random.randn(n, n)
    B = np.ones((n, n))
    
    prev_error = float('inf')
    degeneracy_count = 0
    
    for iteration in range(max_iter):
        # Update B given A (with careful division)
        B_new = np.zeros_like(B)
        mask = np.abs(A) > 1e-10
        B_new[mask] = C[mask] / A[mask]
        
        # Project B to rank-2 with degeneracy handling
        try:
            params_B = parameterize_rank2_robust(B_new)
            if params_B['condition'] > 1e10:
                degeneracy_count += 1
                if verbose and degeneracy_count % 10 == 1:
                    print(f"  Warning: Near-degenerate B at iteration {iteration}")
                # Perturb
                B_new += 0.01 * np.random.randn(n, n)
            
            U_B, s_B, Vt_B = svd(B_new)
            s_B[2:] = 0
            B = U_B @ np.diag(s_B) @ Vt_B
            
        except:
            # Fallback to simple projection
            U_B, s_B, Vt_B = svd(B_new)
            s_B[2:] = 0
            B = U_B @ np.diag(s_B) @ Vt_B
        
        # Update A given B
        A_new = np.zeros_like(A)
        mask = np.abs(B) > 1e-10
        A_new[mask] = C[mask] / B[mask]
        
        # Project A to rank-2
        try:
            params_A = parameterize_rank2_robust(A_new)
            if params_A['condition'] > 1e10:
                degeneracy_count += 1
                A_new += 0.01 * np.random.randn(n, n)
            
            U_A, s_A, Vt_A = svd(A_new)
            s_A[2:] = 0
            A = U_A @ np.diag(s_A) @ Vt_A
            
        except:
            U_A, s_A, Vt_A = svd(A_new)
            s_A[2:] = 0
            A = U_A @ np.diag(s_A) @ Vt_A
        
        # Check convergence
        error = np.linalg.norm(A * B - C, 'fro')
        
        if abs(error - prev_error) < tol or error < tol:
            break
            
        prev_error = error
    
    # Final parameterization for certificate
    params_A = parameterize_rank2_robust(A)
    params_B = parameterize_rank2_robust(B)
    
    return A, B, error, {
        'iterations': iteration,
        'degeneracy_events': degeneracy_count,
        'A_condition': params_A['condition'] if params_A else float('inf'),
        'B_condition': params_B['condition'] if params_B else float('inf'),
        'A_basis_rows': params_A['row_indices'] if params_A else None,
        'B_basis_rows': params_B['row_indices'] if params_B else None
    }

def verify_factorization(C, A, B, tol=1e-10):
    """
    Verify the factorization and provide a certificate.
    """
    # Check error
    error = np.linalg.norm(A * B - C, 'fro')
    rel_error = error / np.linalg.norm(C, 'fro')
    
    # Check ranks
    rank_A = np.linalg.matrix_rank(A, tol=tol)
    rank_B = np.linalg.matrix_rank(B, tol=tol)
    
    # Get robust parameterizations
    params_A = parameterize_rank2_robust(A)
    params_B = parameterize_rank2_robust(B)
    
    certificate = {
        'valid': rel_error < tol and rank_A <= 2 and rank_B <= 2,
        'relative_error': rel_error,
        'rank_A': rank_A,
        'rank_B': rank_B,
        'A_basis': {
            'rows_used': params_A['row_indices'] if params_A else None,
            'condition': params_A['condition'] if params_A else float('inf'),
            'basis_vectors': params_A['basis'].tolist() if params_A else None
        },
        'B_basis': {
            'rows_used': params_B['row_indices'] if params_B else None,
            'condition': params_B['condition'] if params_B else float('inf'),
            'basis_vectors': params_B['basis'].tolist() if params_B else None
        }
    }
    
    return certificate

def test_certified_solver():
    """Test the certified solver on various cases."""
    print("CERTIFIED HADAMARD FACTORIZATION")
    print("="*60)
    
    np.random.seed(42)
    
    # Test 1: Well-conditioned case
    print("\nTest 1: Well-conditioned rank-2 factorization")
    A_true = np.random.randn(4, 2) @ np.random.randn(2, 4)
    B_true = np.random.randn(4, 2) @ np.random.randn(2, 4)
    C1 = A_true * B_true
    
    A, B, error, info = solve_hadamard_certified(C1, verbose=False)
    cert = verify_factorization(C1, A, B)
    
    print(f"Valid: {cert['valid']}")
    print(f"Relative error: {cert['relative_error']:.6e}")
    print(f"A basis uses rows: {cert['A_basis']['rows_used']}, condition: {cert['A_basis']['condition']:.2e}")
    print(f"B basis uses rows: {cert['B_basis']['rows_used']}, condition: {cert['B_basis']['condition']:.2e}")
    
    # Test 2: Near-degenerate case
    print("\nTest 2: Near-degenerate case (rows nearly collinear)")
    v1 = np.array([1, 2, 3, 4])
    v2 = v1 + 0.001 * np.random.randn(4)  # Nearly collinear
    A_bad = np.array([v1, v2, 2*v1 + 3*v2, v1 - v2])
    B_bad = np.random.randn(4, 2) @ np.random.randn(2, 4)
    C2 = A_bad * B_bad
    
    print(f"Condition number of first two rows: {np.linalg.cond(np.array([v1, v2])):.2e}")
    
    A, B, error, info = solve_hadamard_certified(C2, verbose=True)
    cert = verify_factorization(C2, A, B)
    
    print(f"Valid: {cert['valid']}")
    print(f"Relative error: {cert['relative_error']:.6e}")
    print(f"Degeneracy events during optimization: {info['degeneracy_events']}")
    print(f"A basis uses rows: {cert['A_basis']['rows_used']}")
    print(f"B basis uses rows: {cert['B_basis']['rows_used']}")
    
    # Test 3: Random matrices with robustness check
    print("\nTest 3: Random matrices with condition monitoring")
    n_tests = 10
    results = []
    
    for i in range(n_tests):
        C = np.random.randn(4, 4)
        A, B, error, info = solve_hadamard_certified(C, verbose=False)
        cert = verify_factorization(C, A, B)
        
        results.append({
            'valid': cert['valid'],
            'error': cert['relative_error'],
            'A_cond': info['A_condition'],
            'B_cond': info['B_condition'],
            'degeneracies': info['degeneracy_events']
        })
    
    valid_count = sum(r['valid'] for r in results)
    avg_A_cond = np.mean([r['A_cond'] for r in results if r['A_cond'] < 1e10])
    avg_B_cond = np.mean([r['B_cond'] for r in results if r['B_cond'] < 1e10])
    
    print(f"\nSuccess rate: {valid_count}/{n_tests}")
    print(f"Average condition numbers: A={avg_A_cond:.2e}, B={avg_B_cond:.2e}")
    print(f"Total degeneracy events: {sum(r['degeneracies'] for r in results)}")
    
    # Test 4: Stress test with rank-1 matrix
    print("\nTest 4: Rank-1 matrix (should fail gracefully)")
    v = np.random.randn(4)
    C4 = np.outer(v, v)
    
    A, B, error, info = solve_hadamard_certified(C4, verbose=False)
    cert = verify_factorization(C4, A, B)
    
    print(f"Input rank: {np.linalg.matrix_rank(C4)}")
    print(f"Valid: {cert['valid']}")
    print(f"Relative error: {cert['relative_error']:.6e}")
    print(f"Output ranks: A={cert['rank_A']}, B={cert['rank_B']}")

if __name__ == "__main__":
    test_certified_solver()