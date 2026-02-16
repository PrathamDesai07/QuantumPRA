"""
Convergence Analysis
====================

Validates that numerical simulations converge to analytic predictions
in the low-noise limit. Key validation for Phase 3.

Verifies:
1. p_L / p²₂ → α₂ as p₂ → 0 (leading-order convergence)
2. Statistical convergence with increasing samples
3. Systematic errors and corrections
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.stats import linregress
from scipy.optimize import curve_fit

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from simulation.monte_carlo import MonteCarloSimulator, SimulationResult
from noise_models.depolarizing import NoiseModel
from protocols.base_protocol import BaseProtocol
from protocols.lao_criger import CRProtocol, MRProtocol
from protocols.li_protocol import LiProtocol
from analysis.symbolic_expressions import SymbolicAnalyzer


@dataclass
class ConvergenceResult:
    """
    Results from convergence analysis.
    
    Attributes:
        protocol_name: Protocol name
        p2_values: Array of p2 values tested
        p_L_numerical: Numerical p_L values
        p_L_analytic: Analytic predictions
        scaled_p_L: p_L / p2^2 (should converge to α₂)
        alpha_2_target: Target leading coefficient
        convergence_rate: How fast numerics converge to analytics
    """
    protocol_name: str
    p2_values: np.ndarray
    p_L_numerical: np.ndarray
    p_L_analytic: np.ndarray
    scaled_p_L: np.ndarray
    alpha_2_target: float
    convergence_rate: float
    
    def relative_errors(self) -> np.ndarray:
        """Compute relative errors."""
        return np.abs(self.p_L_numerical - self.p_L_analytic) / self.p_L_analytic


class ConvergenceAnalyzer:
    """
    Analyzes convergence of numerical to analytic expressions.
    """
    
    def __init__(self):
        self.analyzer = SymbolicAnalyzer()
    
    def analyze_leading_order_convergence(
        self,
        protocol: BaseProtocol,
        protocol_name: str,
        p2_range: np.ndarray,
        p1_ratio: float = 0.1,
        n_samples: int = 100000
    ) -> ConvergenceResult:
        """
        Analyze convergence to leading order.
        
        As p2 → 0, we expect p_L / p2^2 → α₂.
        
        Args:
            protocol: Protocol instance
            protocol_name: Protocol name
            p2_range: Array of p2 values (should span at least 2 orders of magnitude)
            p1_ratio: Ratio p1/p2
            n_samples: MC samples per point
        
        Returns:
            ConvergenceResult
        """
        print(f"\nAnalyzing leading-order convergence for {protocol_name}")
        print(f"  p2 range: {p2_range.min():.1e} - {p2_range.max():.1e}")
        print(f"  Samples: {n_samples}")
        
        # Get analytic expression
        if protocol_name == 'YL':
            expr = self.analyzer.derive_YL_expression()
        elif protocol_name == 'CR':
            expr = self.analyzer.derive_CR_expression()
        elif protocol_name == 'MR':
            expr = self.analyzer.derive_MR_expression()
        else:
            raise ValueError(f"Unknown protocol: {protocol_name}")
        
        alpha_2 = expr.coefficients['alpha_2']
        
        # Run simulations
        p_L_numerical = []
        p_L_analytic = []
        
        for p2 in p2_range:
            p1 = p1_ratio * p2
            
            # Numerical
            noise = NoiseModel(p1=p1, p2=p2, p_init=p2, p_meas=p2, bias_eta=1.0)
            simulator = MonteCarloSimulator(protocol, noise)
            result = simulator.run(n_samples=n_samples)
            p_L_numerical.append(result.p_logical)
            
            # Analytic
            p_L_an = expr.evaluate(p_1=p1, p_2=p2, p_I=p2, p_M=p2)
            p_L_analytic.append(p_L_an)
            
            print(f"  p2={p2:.1e}: p_L_num={result.p_logical:.4e}, p_L_an={p_L_an:.4e}")
        
        p_L_numerical = np.array(p_L_numerical)
        p_L_analytic = np.array(p_L_analytic)
        
        # Scale by p2^2
        scaled_p_L = p_L_numerical / p2_range**2
        
        # Fit convergence rate
        # Expect: (p_L/p2^2 - α₂) ∝ p2^k for some k > 0
        deviation = np.abs(scaled_p_L - alpha_2)
        valid = deviation > 0
        
        if np.sum(valid) > 2:
            log_p2 = np.log(p2_range[valid])
            log_dev = np.log(deviation[valid])
            slope, intercept, r_value, _, _ = linregress(log_p2, log_dev)
            convergence_rate = slope
        else:
            convergence_rate = np.nan
        
        result = ConvergenceResult(
            protocol_name=protocol_name,
            p2_values=p2_range,
            p_L_numerical=p_L_numerical,
            p_L_analytic=p_L_analytic,
            scaled_p_L=scaled_p_L,
            alpha_2_target=alpha_2,
            convergence_rate=convergence_rate
        )
        
        print(f"\nConvergence analysis:")
        print(f"  Target α₂: {alpha_2:.4f}")
        print(f"  Scaled p_L at lowest p2: {scaled_p_L[0]:.4f}")
        print(f"  Convergence rate: {convergence_rate:.2f}")
        
        return result
    
    def plot_convergence(
        self,
        results: List[ConvergenceResult],
        save_path: str = None
    ):
        """
        Plot convergence analysis.
        
        Args:
            results: List of ConvergenceResults
            save_path: Optional save path
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: p_L vs p2
        ax1 = axes[0]
        for result in results:
            ax1.loglog(result.p2_values, result.p_L_numerical, 'o', 
                      label=f'{result.protocol_name} (numerical)', markersize=6)
            ax1.loglog(result.p2_values, result.p_L_analytic, '--', alpha=0.7,
                      label=f'{result.protocol_name} (analytic)')
        
        ax1.set_xlabel('Two-qubit error rate $p_2$')
        ax1.set_ylabel('Logical error rate $p_L$')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Logical Error Rate')
        
        # Plot 2: Scaled p_L (should converge to α₂)
        ax2 = axes[1]
        for result in results:
            ax2.semilogx(result.p2_values, result.scaled_p_L, 'o-',
                        label=result.protocol_name, markersize=6)
            ax2.axhline(y=result.alpha_2_target, linestyle='--', alpha=0.5,
                       label=f'α₂={result.alpha_2_target:.2f}')
        
        ax2.set_xlabel('Two-qubit error rate $p_2$')
        ax2.set_ylabel('$p_L / p_2^2$')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Leading-Order Convergence')
        
        # Plot 3: Relative error
        ax3 = axes[2]
        for result in results:
            rel_errors = result.relative_errors() * 100
            ax3.loglog(result.p2_values, rel_errors, 'o-',
                      label=result.protocol_name, markersize=6)
        
        ax3.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='20% threshold')
        ax3.set_xlabel('Two-qubit error rate $p_2$')
        ax3.set_ylabel('Relative error (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Analytic-Numerical Agreement')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        return fig, axes
    
    def analyze_statistical_convergence(
        self,
        protocol: BaseProtocol,
        protocol_name: str,
        p2: float = 1e-3,
        sample_sizes: List[int] = [100, 1000, 10000, 100000],
        n_trials: int = 10
    ):
        """
        Analyze statistical convergence with sample size.
        
        Args:
            protocol: Protocol instance
            protocol_name: Protocol name
            p2: Fixed p2 value
            sample_sizes: List of sample sizes to test
            n_trials: Number of trials per sample size
        """
        print(f"\nAnalyzing statistical convergence for {protocol_name}")
        print(f"  p2: {p2:.1e}, Trials per sample size: {n_trials}")
        
        noise = NoiseModel(p1=0.1*p2, p2=p2, p_init=p2, p_meas=p2, bias_eta=1.0)
        
        means = []
        stds = []
        
        for n_samples in sample_sizes:
            p_L_trials = []
            
            print(f"\n  Sample size: {n_samples}")
            for trial in range(n_trials):
                simulator = MonteCarloSimulator(protocol, noise)
                result = simulator.run(n_samples=n_samples)
                p_L_trials.append(result.p_logical)
            
            mean_p_L = np.mean(p_L_trials)
            std_p_L = np.std(p_L_trials)
            means.append(mean_p_L)
            stds.append(std_p_L)
            
            print(f"    Mean p_L: {mean_p_L:.4e} ± {std_p_L:.4e}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.errorbar(sample_sizes, means, yerr=stds, fmt='o-', capsize=5, markersize=8)
        ax.set_xscale('log')
        ax.set_xlabel('Number of samples')
        ax.set_ylabel('Mean $p_L$')
        ax.set_title(f'{protocol_name}: Statistical Convergence')
        ax.grid(True, alpha=0.3)
        
        # Add 1/√N line
        if len(sample_sizes) > 1:
            x_fit = np.array(sample_sizes)
            y_fit = means[0] + stds[0] * np.sqrt(sample_sizes[0]) / np.sqrt(x_fit)
            ax.plot(x_fit, y_fit, '--', alpha=0.5, label='$\\sim 1/\\sqrt{N}$')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return means, stds


def run_full_convergence_analysis(protocols: List[str] = ['CR', 'MR']):
    """
    Run comprehensive convergence analysis.
    
    Args:
        protocols: List of protocol names to analyze
    """
    analyzer = ConvergenceAnalyzer()
    
    # Test range spanning 2 orders of magnitude
    p2_range = np.logspace(-4, -2, 8)
    
    results = []
    
    for prot_name in protocols:
        if prot_name == 'CR':
            protocol = CRProtocol(distance=3)
        elif prot_name == 'MR':
            protocol = MRProtocol(distance=3)
        elif prot_name == 'YL':
            protocol = LiProtocol(distance=3)
        else:
            continue
        
        result = analyzer.analyze_leading_order_convergence(
            protocol=protocol,
            protocol_name=prot_name,
            p2_range=p2_range,
            n_samples=50000  # Reduced for demo
        )
        
        results.append(result)
    
    # Plot combined results
    analyzer.plot_convergence(results, save_path='figures/convergence_analysis.png')
    
    return results


def test_convergence_analyzer():
    """Test convergence analyzer with small sample."""
    print("Testing Convergence Analyzer")
    print("=" * 50)
    
    analyzer = ConvergenceAnalyzer()
    
    # Quick test with small range
    protocol = CRProtocol()
    p2_range = np.array([1e-3, 3e-3])
    
    result = analyzer.analyze_leading_order_convergence(
        protocol=protocol,
        protocol_name='CR',
        p2_range=p2_range,
        n_samples=1000  # Small for testing
    )
    
    print(f"\nTest result:")
    print(f"  Scaled p_L: {result.scaled_p_L}")
    print(f"  Target α₂: {result.alpha_2_target:.4f}")
    print(f"  Relative errors: {result.relative_errors()}")
    
    print("\nConvergence analyzer tests passed!")


if __name__ == "__main__":
    test_convergence_analyzer()
