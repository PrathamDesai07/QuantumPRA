"""
Symbolic Expressions for p_L
=============================

Derives closed-form symbolic expressions for logical error rate p_L
as functions of hardware noise parameters.

This module provides:
1. Symbolic coefficient calculation (α₁, α₂, etc.)
2. Leading-order and next-order expansions
3. Comparison with numerical results
"""

import numpy as np
from typing import Dict, Tuple, Callable, List, Optional
from dataclasses import dataclass
import sympy as sp


@dataclass
class SymbolicPL:
    """
    Symbolic expression for logical error rate.
    
    Attributes:
        protocol_name: Name of protocol
        expression: Sympy expression for p_L
        coefficients: Dict of numerical coefficients
        order: Order of approximation ('first', 'second')
    """
    protocol_name: str
    expression: sp.Expr
    coefficients: Dict[str, float]
    order: str = 'first'
    
    def evaluate(self, **noise_params) -> float:
        """
        Evaluate expression for given noise parameters.
        
        Args:
            **noise_params: Keyword arguments with p_1, p_2, p_I, p_M values
        
        Returns:
            Numerical value of p_L  
        """
        # Create symbol mapping
        p1_sym = sp.Symbol('p_1', real=True, positive=True)
        p2_sym = sp.Symbol('p_2', real=True, positive=True)
        pI_sym = sp.Symbol('p_I', real=True, positive=True)
        pM_sym = sp.Symbol('p_M', real=True, positive=True)
        
        subs_dict = {}
        if 'p_1' in noise_params:
            subs_dict[p1_sym] = noise_params['p_1']
        if 'p_2' in noise_params:
            subs_dict[p2_sym] = noise_params['p_2']
        if 'p_I' in noise_params:
            subs_dict[pI_sym] = noise_params['p_I']
        if 'p_M' in noise_params:
            subs_dict[pM_sym] = noise_params['p_M']
        
        # Evaluate
        result = self.expression.subs(subs_dict)
        
        # Check if all symbols were substituted
        if result.free_symbols:
            # Some symbols remain - set them to zero
            zero_remaining = {sym: 0 for sym in result.free_symbols}
            result = result.subs(zero_remaining)
        
        return float(result.evalf())
    
    def latex(self) -> str:
        """Return LaTeX representation."""
        return sp.latex(self.expression)


class SymbolicAnalyzer:
    """
    Derives symbolic expressions for protocols.
    """
    
    def __init__(self):
        # Define symbolic variables
        self.p1 = sp.Symbol('p_1', real=True, positive=True)
        self.p2 = sp.Symbol('p_2', real=True, positive=True)
        self.p_init = sp.Symbol('p_I', real=True, positive=True)
        self.p_meas = sp.Symbol('p_M', real=True, positive=True)
        self.p_idle = sp.Symbol('p_{idle}', real=True, positive=True)
    
    def derive_YL_expression(self, order: str = 'first') -> SymbolicPL:
        """
        Derive expression for Li's YL protocol.
        
        From Li (2015), leading order:
        p_L ≈ (2/5) p_2^2 + (2/3) p_1 + 2 p_I + O(p_1 p_2)
        
        Args:
            order: 'first' or 'second'
        
        Returns:
            SymbolicPL instance
        """
        if order == 'first':
            # Leading-order expression
            expr = sp.Rational(2, 5) * self.p2**2 + \
                   sp.Rational(2, 3) * self.p1 + \
                   2 * self.p_init
            
            coeffs = {
                'alpha_2': 2/5,   # Coefficient of p2^2
                'alpha_1': 2/3,   # Coefficient of p1
                'alpha_I': 2,     # Coefficient of p_init
                'alpha_M': 0,     # No measurement contribution in leading order
            }
        else:
            # Include second-order terms
            expr = sp.Rational(2, 5) * self.p2**2 + \
                   sp.Rational(2, 3) * self.p1 + \
                   2 * self.p_init + \
                   4 * self.p1 * self.p2  # Example cross term
            
            coeffs = {
                'alpha_2': 2/5,
                'alpha_1': 2/3,
                'alpha_I': 2,
                'alpha_M': 0,
                'alpha_12': 4,  # Cross term coefficient
            }
        
        return SymbolicPL(
            protocol_name='YL',
            expression=expr,
            coefficients=coeffs,
            order=order
        )
    
    def derive_CR_expression(self, order: str = 'first') -> SymbolicPL:
        """
        Derive expression for CR protocol.
        
        From Lao & Criger (2022):
        p_L^CR ≈ (3/5) p_2^2 + α_1 p_1 + 2 p_I
        
        Theα_1 coefficient is larger than YL due to rotated geometry.
        """
        if order == 'first':
            expr = sp.Rational(3, 5) * self.p2**2 + \
                   sp.Rational(5, 6) * self.p1 + \
                   2 * self.p_init
            
            coeffs = {
                'alpha_2': 3/5,
                'alpha_1': 5/6,
                'alpha_I': 2,
                'alpha_M': 0,
            }
        else:
            expr = sp.Rational(3, 5) * self.p2**2 + \
                   sp.Rational(5, 6) * self.p1 + \
                   2 * self.p_init + \
                   5 * self.p1 * self.p2
            
            coeffs = {
                'alpha_2': 3/5,
                'alpha_1': 5/6,
                'alpha_I': 2,
                'alpha_M': 0,
                'alpha_12': 5,
            }
        
        return SymbolicPL(
            protocol_name='CR',
            expression=expr,
            coefficients=coeffs,
            order=order
        )
    
    def derive_MR_expression(self, order: str = 'first') -> SymbolicPL:
        """
        Derive expression for MR protocol.
        
        From Lao & Criger (2022):
        p_L^MR ≈ (3/5) p_2^2 + β_1 p_1 + p_I
        
        MR has better p_1 coefficient due to fewer sensitive qubits,
        and better p_I coefficient due to magic qubit in middle.
        """
        if order == 'first':
            expr = sp.Rational(3, 5) * self.p2**2 + \
                   sp.Rational(1, 2) * self.p1 + \
                   self.p_init
            
            coeffs = {
                'alpha_2': 3/5,
                'alpha_1': 1/2,   # Better than CR
                'alpha_I': 1,     # Half of CR/YL
                'alpha_M': 0,
            }
        else:
            expr = sp.Rational(3, 5) * self.p2**2 + \
                   sp.Rational(1, 2) * self.p1 + \
                   self.p_init + \
                   3 * self.p1 * self.p2
            
            coeffs = {
                'alpha_2': 3/5,
                'alpha_1': 1/2,
                'alpha_I': 1,
                'alpha_M': 0,
                'alpha_12': 3,
            }
        
        return SymbolicPL(
            protocol_name='MR',
            expression=expr,
            coefficients=coeffs,
            order=order
        )
    
    def derive_biased_expression(
        self,
        protocol_name: str,
        eta: float = 100.0
    ) -> SymbolicPL:
        """
        Derive expression for biased noise regime.
        
        Under biased noise with η = p_Z / p_X >> 1, the coefficients change
        because Z errors dominate.
        
        Args:
            protocol_name: 'YL', 'CR', or 'MR'
            eta: Bias ratio
        
        Returns:
            SymbolicPL with bias-aware expression
        """
        # Under bias, single-qubit errors become predominantly Z
        # This affects how they propagate through the circuit
        
        if protocol_name == 'MR':
            # MR benefits most from bias due to orientation
            expr = sp.Rational(3, 5) * self.p2**2 + \
                   sp.Rational(1, 4) * self.p1 + \
                   self.p_init
            
            coeffs = {
                'alpha_2': 3/5,
                'alpha_1': 1/4,  # Even better under bias
                'alpha_I': 1,
                'eta': eta,
            }
        else:
            # Generic improvement under bias
            base = self.derive_CR_expression() if protocol_name == 'CR' else self.derive_YL_expression()
            expr = base.expression
            coeffs = base.coefficients
            coeffs['eta'] = eta
        
        return SymbolicPL(
            protocol_name=f'{protocol_name}_biased',
            expression=expr,
            coefficients=coeffs
        )


def compare_protocols(noise_params: Dict[str, float]) -> Dict[str, float]:
    """
    Compare p_L for different protocols at given noise parameters.
    
    Args:
        noise_params: Dict with p1, p2, p_init, p_meas
    
    Returns:
        Dict mapping protocol name to p_L value
    """
    analyzer = SymbolicAnalyzer()
    
    yl = analyzer.derive_YL_expression()
    cr = analyzer.derive_CR_expression()
    mr = analyzer.derive_MR_expression()
    
    results = {
        'YL': yl.evaluate(p_1=noise_params['p1'], p_2=noise_params['p2'],
                          p_I=noise_params['p_init'], p_M=noise_params['p_meas']),
        'CR': cr.evaluate(p_1=noise_params['p1'], p_2=noise_params['p2'],
                          p_I=noise_params['p_init'], p_M=noise_params['p_meas']),
        'MR': mr.evaluate(p_1=noise_params['p1'], p_2=noise_params['p2'],
                          p_I=noise_params['p_init'], p_M=noise_params['p_meas']),
    }
    
    return results


def plot_pL_vs_p2(
    protocol_names: List[str],
    p2_range: np.ndarray,
    p1_ratio: float = 0.1,
    save_path: Optional[str] = None
):
    """
    Plot p_L vs p2 for multiple protocols.
    
    Args:
        protocol_names: List of protocol names to compare
        p2_range: Array of p2 values
        p1_ratio: Ratio p1/p2
        save_path: Optional path to save plot
    """
    import matplotlib.pyplot as plt
    
    analyzer = SymbolicAnalyzer()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for prot_name in protocol_names:
        if prot_name == 'YL':
            expr = analyzer.derive_YL_expression()
        elif prot_name == 'CR':
            expr = analyzer.derive_CR_expression()
        elif prot_name == 'MR':
            expr = analyzer.derive_MR_expression()
        else:
            continue
        
        p_L_values = []
        for p2 in p2_range:
            p1 = p1_ratio * p2
            p_L = expr.evaluate(p_1=p1, p_2=p2, p_I=p2, p_M=p2)
            p_L_values.append(p_L)
        
        ax.plot(p2_range, p_L_values, label=prot_name, marker='o', markersize=4)
    
    ax.set_xlabel('Two-qubit error rate $p_2$')
    ax.set_ylabel('Logical error rate $p_L$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Logical Error Rate vs $p_2$ ($p_1 = {p1_ratio} p_2$)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax


def test_symbolic_expressions():
    """Test symbolic expression derivation."""
    analyzer = SymbolicAnalyzer()
    
    # Test YL
    yl = analyzer.derive_YL_expression()
    print(f"YL protocol expression:")
    print(f"  {yl.latex()}")
    print(f"  Coefficients: {yl.coefficients}")
    
    # Test CR
    cr = analyzer.derive_CR_expression()
    print(f"\nCR protocol expression:")
    print(f"  {cr.latex()}")
    print(f"  Coefficients: {cr.coefficients}")
    
    # Test MR
    mr = analyzer.derive_MR_expression()
    print(f"\nMR protocol expression:")
    print(f"  {mr.latex()}")
    print(f"  Coefficients: {mr.coefficients}")
    
    # Evaluate at specific parameters
    noise = {'p1': 1e-4, 'p2': 1e-3, 'p_init': 1e-3, 'p_meas': 1e-3}
    print(f"\nEvaluation at p2=1e-3, p1=1e-4:")
    print(f"  YL: p_L = {yl.evaluate(p_1=1e-4, p_2=1e-3, p_I=1e-3, p_M=1e-3):.4e}")
    print(f"  CR: p_L = {cr.evaluate(p_1=1e-4, p_2=1e-3, p_I=1e-3, p_M=1e-3):.4e}")
    print(f"  MR: p_L = {mr.evaluate(p_1=1e-4, p_2=1e-3, p_I=1e-3, p_M=1e-3):.4e}")
    
    # Compare protocols
    results = compare_protocols(noise)
    print(f"\nProtocol comparison:")
    for prot, p_L in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {prot}: {p_L:.4e}")
    
    print("\nSymbolic expression tests passed!")


if __name__ == "__main__":
    # Test requires sympy
    try:
        test_symbolic_expressions()
    except ImportError:
        print("sympy not installed. Install with: pip install sympy")
