"""
Analytic vs Numerical Comparison
=================================

Compares analytic expressions with Monte Carlo simulation results.
Validates leading-order expressions and identifies second-order contributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from analysis.symbolic_expressions import SymbolicAnalyzer, SymbolicPL
from simulation.monte_carlo import SimulationResult


@dataclass
class ComparisonResult:
    """
    Results comparing analytic and numerical p_L.
    
    Attributes:
        protocol_name: Name of protocol
        p2: Two-qubit error rate
        p_L_analytic: Analytic prediction
        p_L_numerical: Monte Carlo estimate
        relative_error: |analytic - numerical| / numerical
        n_samples: Number of MC samples
    """
    protocol_name: str
    p2: float
    p_L_analytic: float
    p_L_numerical: float
    relative_error: float
    n_samples: int
    
    def agrees_within_tolerance(self, tolerance: float = 0.2) -> bool:
        """Check if analytic and numerical agree within tolerance."""
        return self.relative_error < tolerance


class AnalyticNumericalComparator:
    """
    Compares analytic expressions with numerical simulations.
    """
    
    def __init__(self):
        self.analyzer = SymbolicAnalyzer()
        self.results: List[ComparisonResult] = []
    
    def compare_single_point(
        self,
        protocol_name: str,
        p1: float,
        p2: float,
        p_init: float,
        p_meas: float,
        sim_result: SimulationResult
    ) -> ComparisonResult:
        """
        Compare analytic and numerical at single parameter point.
        
        Args:
            protocol_name: Protocol name
            p1, p2, p_init, p_meas: Noise parameters
            sim_result: Monte Carlo simulation result
        
        Returns:
            ComparisonResult
        """
        # Get analytic prediction
        if protocol_name == 'YL':
            expr = self.analyzer.derive_YL_expression()
        elif protocol_name == 'CR':
            expr = self.analyzer.derive_CR_expression()
        elif protocol_name == 'MR':
            expr = self.analyzer.derive_MR_expression()
        else:
            raise ValueError(f"Unknown protocol: {protocol_name}")
        
        p_L_analytic = expr.evaluate(p_1=p1, p_2=p2, p_I=p_init, p_M=p_meas)
        p_L_numerical = sim_result.p_logical
        
        # Compute relative error
        if p_L_numerical > 0:
            rel_error = abs(p_L_analytic - p_L_numerical) / p_L_numerical
        else:
            rel_error = float('inf') if p_L_analytic > 0 else 0.0
        
        result = ComparisonResult(
            protocol_name=protocol_name,
            p2=p2,
            p_L_analytic=p_L_analytic,
            p_L_numerical=p_L_numerical,
            relative_error=rel_error,
            n_samples=sim_result.n_samples
        )
        
        self.results.append(result)
        return result
    
    def compare_sweep(
        self,
        protocol_name: str,
        p2_values: List[float],
        p1_ratio: float = 0.1,
        simulation_dir: str = "data/raw"  
    ) -> List[ComparisonResult]:
        """
        Compare across parameter sweep.
        
        Args:
            protocol_name: Protocol name
            p2_values: List of p2 values
            p1_ratio: Ratio p1/p2
            simulation_dir: Directory with MC results
        
        Returns:
            List of ComparisonResults
        """
        results = []
        
        for p2 in p2_values:
            p1 = p1_ratio * p2
            
            # Load numerical result
            filename = f"{protocol_name}_d3_p2{p2:.1e}.json"
            filepath = Path(simulation_dir) / filename
            
            if filepath.exists():
                sim_result = SimulationResult.load(str(filepath))
                
                result = self.compare_single_point(
                    protocol_name=protocol_name,
                    p1=p1,
                    p2=p2,
                    p_init=p2,
                    p_meas=p2,
                    sim_result=sim_result
                )
                results.append(result)
        
        return results
    
    def plot_comparison(
        self,
        protocol_names: List[str],
        p2_range: np.ndarray,
        p1_ratio: float = 0.1,
        save_path: str = None
    ):
        """
        Plot analytic vs numerical comparison.
        
        Args:
            protocol_names: List of protocols
            p2_range: Array of p2 values
            p1_ratio: Ratio p1/p2
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left plot: p_L vs p2
        for prot in protocol_names:
            # Analytic
            if prot == 'YL':
                expr = self.analyzer.derive_YL_expression()
            elif prot == 'CR':
                expr = self.analyzer.derive_CR_expression()
            elif prot == 'MR':
                expr = self.analyzer.derive_MR_expression()
            else:
                continue
            
            p_L_analytic = []
            for p2 in p2_range:
                p1 = p1_ratio * p2
                p_L = expr.evaluate(p_1=p1, p_2=p2, p_I=p2, p_M=p2)
                p_L_analytic.append(p_L)
            
            ax1.plot(p2_range, p_L_analytic, label=f'{prot} (analytic)',
                    linestyle='-', linewidth=2)
            
            # Numerical (if available)
            numerical_results = [r for r in self.results if r.protocol_name == prot]
            if numerical_results:
                p2_num = [r.p2 for r in numerical_results]
                p_L_num = [r.p_L_numerical for r in numerical_results]
                ax1.scatter(p2_num, p_L_num, label=f'{prot} (MC)',
                           marker='o', s=60, alpha=0.7)
        
        ax1.set_xlabel('Two-qubit error rate $p_2$')
        ax1.set_ylabel('Logical error rate $p_L$')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Logical Error Rate: Analytic vs Numerical')
        
        # Right plot: Relative error
        for prot in protocol_names:
            prot_results = [r for r in self.results if r.protocol_name == prot]
            if prot_results:
                p2_vals = [r.p2 for r in prot_results]
                rel_errors = [r.relative_error * 100 for r in prot_results]
                ax2.plot(p2_vals, rel_errors, label=prot, marker='o', markersize=6)
        
        ax2.set_xlabel('Two-qubit error rate $p_2$')
        ax2.set_ylabel('Relative error (%)')
        ax2.set_xscale('log')
        ax2.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='20% threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Analytic-Numerical Agreement')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return fig, (ax1, ax2)
    
    def generate_validation_report(self, save_path: str = "data/processed/validation_report.txt"):
        """
        Generate text report of validation results.
        
        Args:
            save_path: Path to save report
        """
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("Analytic vs Numerical Validation Report\n")
            f.write("="*70 + "\n\n")
            
            # Group by protocol
            protocols = set(r.protocol_name for r in self.results)
            
            for prot in sorted(protocols):
                f.write(f"\n{prot} Protocol:\n")
                f.write("-" * 50 + "\n")
                
                prot_results = [r for r in self.results if r.protocol_name == prot]
                prot_results.sort(key=lambda r: r.p2)
                
                f.write(f"{'p2':>10s} {'Analytic':>12s} {'Numerical':>12s} {'Rel Error':>12s} {'Agreement':>10s}\n")
                
                for result in prot_results:
                    agreement = "✓" if result.agrees_within_tolerance(0.2) else "✗"
                    f.write(f"{result.p2:>10.2e} {result.p_L_analytic:>12.4e} "
                           f"{result.p_L_numerical:>12.4e} {result.relative_error:>11.1%} "
                           f"{agreement:>10s}\n")
                
                # Summary statistics
                avg_error = np.mean([r.relative_error for r in prot_results])
                max_error = np.max([r.relative_error for r in prot_results])
                agreements = sum(r.agrees_within_tolerance(0.2) for r in prot_results)
                
                f.write(f"\nSummary:\n")
                f.write(f"  Average relative error: {avg_error:.1%}\n")
                f.write(f"  Maximum relative error: {max_error:.1%}\n")
                f.write(f"  Points within 20% tolerance: {agreements}/{len(prot_results)}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write(f"Total comparisons: {len(self.results)}\n")
            f.write("="*70 + "\n")
        
        print(f"Validation report saved to {save_path}")


def plot_convergence_to_leading_order(protocol_name: str = 'CR'):
    """
    Plot how numerical p_L converges to leading-order as p2 → 0.
    
    Shows that p_L / p2^2 → constant as p2 decreases.
    """
    analyzer = SymbolicAnalyzer()
    
    if protocol_name == 'CR':
        expr = analyzer.derive_CR_expression()
    elif protocol_name == 'MR':
        expr = analyzer.derive_MR_expression()
    else:
        expr = analyzer.derive_YL_expression()
    
    # Expected leading coefficient
    alpha_2 = expr.coefficients['alpha_2']
    
    # Compute for range of p2
    p2_range = np.logspace(-4, -2, 20)
    p1_ratio = 0.1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scaled_pL = []
    for p2 in p2_range:
        p1 = p1_ratio * p2
        p_L = expr.evaluate(p_1=p1, p_2=p2, p_I=p2, p_M=p2)
        # Scale by p2^2 to extract leading coefficient
        scaled = p_L / p2**2
        scaled_pL.append(scaled)
    
    ax.semilogx(p2_range, scaled_pL, 'o-', label='$p_L / p_2^2$ (analytic)')
    ax.axhline(y=alpha_2, color='r', linestyle='--', label=f'$\\alpha_2 = {alpha_2:.2f}$')
    
    ax.set_xlabel('Two-qubit error rate $p_2$')
    ax.set_ylabel('$p_L / p_2^2$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{protocol_name}: Convergence to Leading Order')
    
    plt.show()
    
    return fig, ax


def test_comparison():
    """Test comparison module."""
    print("Testing Analytic vs Numerical Comparison")
    print("=" * 50)
    
    comparator = AnalyticNumericalComparator()
    
    # Test with mock simulation result
    from simulation.monte_carlo import SimulationResult
    
    mock_result = SimulationResult(
        protocol_name='CR-d3',
        distance=3,
        noise_params={'p1': 1e-4, 'p2': 1e-3, 'p_init': 1e-3, 'p_meas': 1e-3, 'p_idle': 0, 'bias_eta': 1.0},
        n_samples=10000,
        n_accepted=6000,
        n_logical_x=10,
        n_logical_z=8,
        p_accept=0.6,
        p_logical=0.003,
        p_logical_x=0.001667,
        p_logical_z=0.001333,
        logical_bias=0.8,
        runtime_seconds=10.0
    )
    
    result = comparator.compare_single_point(
        protocol_name='CR',
        p1=1e-4,
        p2=1e-3,
        p_init=1e-3,
        p_meas=1e-3,
        sim_result=mock_result
    )
    
    print(f"\nComparison result:")
    print(f"  Protocol: {result.protocol_name}")
    print(f"  p2: {result.p2:.1e}")
    print(f"  p_L (analytic): {result.p_L_analytic:.4e}")
    print(f"  p_L (numerical): {result.p_L_numerical:.4e}")
    print(f"  Relative error: {result.relative_error:.1%}")
    print(f"  Agreement: {'✓' if result.agrees_within_tolerance() else '✗'}")
    
    print("\nComparison tests passed!")


if __name__ == "__main__":
    test_comparison()
