#!/usr/bin/env python3
"""
Interactive Gradio app for Hadamard factorization.
Users can input a 4x4 matrix and get its factorization (if it exists).
"""

import gradio as gr
import numpy as np
from hadamard_real_demo import solve_hadamard_alternating
from hadamard_real_certified import find_rank2_basis
from hadamard_rigorous_theory import compute_alpha_data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64

def parse_matrix_input(matrix_str):
    """Parse matrix input from text."""
    try:
        # Handle various input formats
        matrix_str = matrix_str.strip()
        
        # Try to parse as rows separated by newlines
        rows = matrix_str.strip().split('\n')
        matrix = []
        
        for row in rows:
            # Remove brackets if present
            row = row.strip().strip('[]')
            # Split by comma or space
            if ',' in row:
                values = [float(x.strip()) for x in row.split(',')]
            else:
                values = [float(x) for x in row.split()]
            matrix.append(values)
        
        matrix = np.array(matrix)
        
        # Validate shape
        if matrix.shape != (4, 4):
            return None, f"Error: Matrix must be 4x4, got {matrix.shape}"
        
        return matrix, None
    
    except Exception as e:
        return None, f"Error parsing matrix: {str(e)}"

def visualize_matrix(matrix, title="Matrix"):
    """Create a visual representation of the matrix."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='equal', vmin=-3, vmax=3)
    
    # Add grid
    for i in range(5):
        ax.axhline(i-0.5, color='black', linewidth=1)
        ax.axvline(i-0.5, color='black', linewidth=1)
    
    # Add values
    for i in range(4):
        for j in range(4):
            val = matrix[i, j]
            color = 'white' if abs(val) > 1.5 else 'black'
            text = f"{val:.3f}" if val != 0 else "0"
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)
    
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    # Convert to base64 for display
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return buf

def factorize_matrix(matrix_input):
    """Main function to factorize a matrix."""
    # Parse input
    C, error_msg = parse_matrix_input(matrix_input)
    if error_msg:
        return error_msg, None, None, None
    
    # Basic matrix info
    info = []
    info.append("📊 **Matrix Analysis**\n")
    info.append(f"- Rank: {np.linalg.matrix_rank(C)}")
    info.append(f"- Determinant: {np.linalg.det(C):.6f}")
    info.append(f"- Condition number: {np.linalg.cond(C):.2e}")
    
    # Count zeros
    n_zeros = np.sum(np.abs(C) < 1e-10)
    info.append(f"- Number of zeros: {n_zeros}")
    
    # Check pattern type
    empty_rows = sum(1 for i in range(4) if all(abs(C[i, j]) < 1e-10 for j in range(4)))
    empty_cols = sum(1 for j in range(4) if all(abs(C[i, j]) < 1e-10 for i in range(4)))
    
    if empty_rows > 0:
        info.append(f"- Empty rows: {empty_rows}")
    if empty_cols > 0:
        info.append(f"- Empty columns: {empty_cols}")
    
    # Try factorization
    info.append("\n🔍 **Factorization Attempt**\n")
    
    try:
        # Multiple attempts with different initializations
        best_error = float('inf')
        best_A, best_B = None, None
        
        for attempt in range(5):
            A, B, error, converged = solve_hadamard_alternating(C, max_iter=500, tol=1e-10)
            
            if error < best_error:
                best_error = error
                best_A, best_B = A, B
        
        rel_error = best_error / (np.linalg.norm(C, 'fro') + 1e-10)
        
        # Check if factorization is successful
        if rel_error < 1e-6 and np.linalg.matrix_rank(best_A) <= 2 and np.linalg.matrix_rank(best_B) <= 2:
            info.append("✅ **Factorization Found!**")
            info.append(f"- Relative error: {rel_error:.2e}")
            info.append(f"- Rank of A: {np.linalg.matrix_rank(best_A)}")
            info.append(f"- Rank of B: {np.linalg.matrix_rank(best_B)}")
            
            # Verify: A ∘ B ≈ C
            reconstructed = best_A * best_B
            verify_error = np.linalg.norm(reconstructed - C, 'fro')
            info.append(f"- Verification error: {verify_error:.2e}")
            
            # Certification
            info.append("\n🏆 **Rigorous Certification**\n")
            alpha_data = compute_alpha_data(best_A, best_B, C)
            
            if alpha_data['certified']:
                info.append("✅ Solution is certified by Smale's α-criterion")
                info.append(f"- α = {alpha_data['alpha']:.6f} < {alpha_data['alpha_threshold']:.6f}")
                info.append(f"- Convergence guaranteed in radius: {alpha_data.get('convergence_radius', 0):.2e}")
            else:
                info.append("⚠️ Solution found but not certified")
                info.append(f"- α = {alpha_data['alpha']:.6f} ≥ {alpha_data['alpha_threshold']:.6f}")
            
            # Find rank-2 bases
            info.append("\n📐 **Rank-2 Decomposition**\n")
            
            basis_A = find_rank2_basis(best_A)
            if basis_A is not None:
                U_A, V_A = basis_A
                info.append(f"A = U_A @ V_A^T with U_A shape {U_A.shape}, V_A shape {V_A.shape}")
            
            basis_B = find_rank2_basis(best_B)
            if basis_B is not None:
                U_B, V_B = basis_B
                info.append(f"B = U_B @ V_B^T with U_B shape {U_B.shape}, V_B shape {V_B.shape}")
            
            # Create visualizations
            fig_A = visualize_matrix(best_A, "Factor A (rank ≤ 2)")
            fig_B = visualize_matrix(best_B, "Factor B (rank ≤ 2)")
            fig_C = visualize_matrix(C, "Original Matrix C")
            
            return "\n".join(info), fig_C, fig_A, fig_B
            
        else:
            info.append("❌ **No Factorization Found**")
            info.append(f"- Best relative error achieved: {rel_error:.2e}")
            
            if rel_error < 1e-3:
                info.append("- Note: Close to factorizable, but doesn't meet strict criteria")
            elif n_zeros >= 3 and n_zeros <= 8:
                info.append("- Note: This matrix is in the 'death valley' (3-8 zeros)")
                info.append("- Such matrices are typically not factorizable unless they have special structure")
            
            # Show why it might have failed
            if np.linalg.matrix_rank(best_A) > 2:
                info.append(f"- Issue: Rank of A = {np.linalg.matrix_rank(best_A)} > 2")
            if np.linalg.matrix_rank(best_B) > 2:
                info.append(f"- Issue: Rank of B = {np.linalg.matrix_rank(best_B)} > 2")
            
            fig_C = visualize_matrix(C, "Original Matrix C")
            
            return "\n".join(info), fig_C, None, None
            
    except Exception as e:
        return f"Error during factorization: {str(e)}", None, None, None

# Example matrices
example_zero_free = """1.5 -0.3 0.8 1.2
-0.3 2.1 0.5 -0.7
0.8 0.5 1.8 0.9
1.2 -0.7 0.9 1.3"""

example_sparse = """1.5 0 0.8 0
0 2.1 0 -0.7
0.8 0 1.8 0
0 -0.7 0 1.3"""

example_row_zeros = """0 0 0 0
1.2 -0.5 0.8 1.1
-0.3 2.1 0.5 -0.7
0.8 0.5 1.8 0.9"""

example_not_factorizable = """1 0 0 1
0 1 1 0
0 1 1 0
1 0 0 1"""

# Create Gradio interface
with gr.Blocks(title="Hadamard Factorization Explorer") as demo:
    gr.Markdown("""
    # 🔢 Hadamard Factorization Explorer
    
    This tool attempts to factorize a 4×4 matrix C as the Hadamard (element-wise) product of two rank-2 matrices:
    
    **C = A ∘ B** where rank(A) ≤ 2 and rank(B) ≤ 2
    
    Enter your matrix below (4×4, space or comma separated values, one row per line).
    """)
    
    with gr.Row():
        with gr.Column():
            matrix_input = gr.Textbox(
                label="Input Matrix (4×4)",
                placeholder="Enter 4x4 matrix, one row per line...",
                lines=4,
                value=example_zero_free
            )
            
            gr.Examples(
                examples=[
                    [example_zero_free, "Zero-free matrix (often factorizable)"],
                    [example_sparse, "Sparse matrix with diagonal pattern"],
                    [example_row_zeros, "Matrix with empty row (usually factorizable)"],
                    [example_not_factorizable, "Likely not factorizable"]
                ],
                inputs=matrix_input,
                label="Example Matrices"
            )
            
            factorize_btn = gr.Button("🚀 Factorize Matrix", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Analysis Results",
                lines=20,
                max_lines=30
            )
    
    with gr.Row():
        plot_C = gr.Image(label="Original Matrix C")
        plot_A = gr.Image(label="Factor A (if found)")
        plot_B = gr.Image(label="Factor B (if found)")
    
    factorize_btn.click(
        fn=factorize_matrix,
        inputs=matrix_input,
        outputs=[output_text, plot_C, plot_A, plot_B]
    )
    
    gr.Markdown("""
    ## 📖 About
    
    This factorization problem arises in various contexts including tensor decomposition and algebraic geometry.
    Our research shows:
    
    - **Zero-free matrices**: ~48% factorizable
    - **Matrices with 3-8 zeros**: Usually not factorizable (the "death valley") unless they have special structure
    - **Matrices with empty rows/columns**: Often factorizable
    - **Very sparse matrices (>10 zeros)**: Increasingly factorizable
    
    The tool uses alternating projection with multiple random initializations and provides rigorous certification
    using Smale's α-criterion when possible.
    """)

if __name__ == "__main__":
    demo.launch(share=True)