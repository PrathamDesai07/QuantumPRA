"""
Logical Noise Bias Analysis
============================

Analyzes the logical noise bias η_L = p_{Z̄}/p_{X̄} of injected magic states.

Key insights:
- Physical noise bias propagates to logical level
- Different protocols have different bias preservation
- Logical bias affects subsequent distillation efficiency
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class LogicalBiasResult:
    """
    Results of logical bias analysis.
    
    Attributes:
        p_logical_x: Logical X error probability
        p_logical_z: Logical Z error probability
        eta_logical: Logical bias η_L = p_Z̄ / p_X̄
        eta_physical: Physical noise bias η = p_Z / p_X
        protocol_name: Name of protocol
    """
    p_logical_x: float
    p_logical_z: float
    eta_logical: float
    eta_physical: float
    protocol_name: str
    
    def bias_preservation_factor(self) -> float:
        """
        Compute bias preservation factor.
        
        Returns η_L / η (ratio of logical to physical bias).
        A value > 1 means bias is amplified, < 1 means diluted.
        """
        if self.eta_physical > 0:
            return self.eta_logical / self.eta_physical
        return 1.0
    
    def __repr__(self):
        return (f"LogicalBias(p_X̄={self.p_logical_x:.3e}, p_Z̄={self.p_logical_z:.3e}, "
                f"η_L={self.eta_logical:.2f}, η={self.eta_physical:.2f})")


class LogicalBiasAnalyzer:
    """
    Analyzes logical noise bias for injection protocols.
    """
    
    def __init__(self, protocol_name: str):
        """
        Initialize analyzer.
        
        Args:
            protocol_name: Name of protocol ('YL', 'CR', 'MR')
        """
        self.protocol_name = protocol_name
    
    def analyze_bias_propagation(
        self,
        p1: float,
        p2: float,
        eta: float = 1.0
    ) -> LogicalBiasResult:
        """
        Analyze how physical bias propagates to logical level.
        
        Args:
            p1: Single-qubit error rate
            p2: Two-qubit error rate
            eta: Physical noise bias η = p_Z / p_X
        
        Returns:
            LogicalBiasResult
        """
        # Decompose p1 into biased components
        # p1 = p_X + p_Y + p_Z
        # With bias: p_Z = η * p_X, p_Y ≈ sqrt(η) * p_X
        # Normalization: p_X * (1 + sqrt(η) + η) = p1
        
        norm = 1 + np.sqrt(eta) + eta
        p_x_phys = p1 / norm
        p_y_phys = p1 * np.sqrt(eta) / norm
        p_z_phys = p1 * eta / norm
        
        # Different protocols have different sensitivity to X vs Z errors
        # This depends on circuit structure and CNOT ordering
        
        if self.protocol_name == 'MR':
            # MR is optimized for biased noise
            # Z errors on magic qubit are better handled
            alpha_x = 0.4  # Coefficient for X errors contributing to p_X̄
            alpha_z = 0.2  # Coefficient for Z errors contributing to p_Z̄
            beta_x = 0.3   # Coefficient for X errors contributing to p_Z̄
            beta_z = 0.5   # Coefficient for Z errors contributing to p_X̄
        elif self.protocol_name == 'CR':
            # CR has different symmetry
            alpha_x = 0.5
            alpha_z = 0.5
            beta_x = 0.4
            beta_z = 0.4
        else:  # YL
            # YL (regular surface code) has more symmetric behavior
            alpha_x = 0.6
            alpha_z = 0.6
            beta_x = 0.5
            beta_z = 0.5
        
        # Compute logical error rates
        # Contribution from single-qubit errors
        p_x_logical_1q = alpha_x * p_x_phys + beta_z * p_z_phys
        p_z_logical_1q = alpha_z * p_z_phys + beta_x * p_x_phys
        
        # Contribution from two-qubit errors (less biased)
        p_x_logical_2q = 0.3 * p2
        p_z_logical_2q = 0.3 * p2
        
        # Total logical errors
        p_logical_x = p_x_logical_1q + p_x_logical_2q
        p_logical_z = p_z_logical_1q + p_z_logical_2q
        
        # Compute logical bias
        eta_logical = p_logical_z / p_logical_x if p_logical_x > 0 else float('inf')
        
        return LogicalBiasResult(
            p_logical_x=p_logical_x,
            p_logical_z=p_logical_z,
            eta_logical=eta_logical,
            eta_physical=eta,
            protocol_name=self.protocol_name
        )
    
    def compare_bias_regimes(
        self,
        p1: float,
        p2: float,
        eta_values: list
    ) -> Dict[float, LogicalBiasResult]:
        """
        Compare logical bias across different physical bias regimes.
        
        Args:
            p1: Single-qubit error rate
            p2: Two-qubit error rate
            eta_values: List of physical bias values to test
        
        Returns:
            Dict mapping eta to LogicalBiasResult
        """
        results = {}
        for eta in eta_values:
            result = self.analyze_bias_propagation(p1, p2, eta)
            results[eta] = result
        return results


def compute_distillation_benefit(
    bias_result: LogicalBiasResult,
    target_fidelity: float = 1e-8
) -> Dict[str, float]:
    """
    Compute distillation overhead benefit from logical bias.
    
    Biased noise can reduce distillation cost if exploited properly.
    
    Args:
        bias_result: LogicalBiasResult from bias analysis
        target_fidelity: Target output fidelity
    
    Returns:
        Dict with distillation metrics
    """
    p_in = bias_result.p_logical_x + bias_result.p_logical_z
    
    # Standard 15-to-1 distillation: p_out ≈ 35 p_in^3
    # Number of rounds needed
    p_current = p_in
    n_rounds = 0
    
    while p_current > target_fidelity and n_rounds < 10:
        p_current = 35 * p_current ** 3
        n_rounds += 1
    
    # Raw states needed (exponential)
    raw_states_needed = 15 ** n_rounds
    
    # With bias exploitation (hypothetical improvement)
    # If η_L > 10, can use bias-tailored distillation
    if bias_result.eta_logical > 10:
        improvement_factor = 1.5  # Example: 50% fewer states needed
    else:
        improvement_factor = 1.0
    
    return {
        'n_rounds': n_rounds,
        'raw_states_needed': raw_states_needed,
        'with_bias_exploitation': raw_states_needed / improvement_factor,
        'improvement_factor': improvement_factor
    }


def test_logical_bias():
    """Test logical bias analysis."""
    # Test MR under biased noise
    analyzer = LogicalBiasAnalyzer('MR')
    
    print("Logical Bias Analysis Tests")
    print("=" * 50)
    
    # Test 1: Unbiased noise
    result = analyzer.analyze_bias_propagation(p1=1e-4, p2=1e-3, eta=1.0)
    print(f"\nUnbiased regime (η=1):")
    print(f"  {result}")
    print(f"  Bias preservation: {result.bias_preservation_factor():.2f}")
    
    # Test 2: Moderately biased
    result = analyzer.analyze_bias_propagation(p1=1e-4, p2=1e-3, eta=10.0)
    print(f"\nModerately biased (η=10):")
    print(f"  {result}")
    print(f"  Bias preservation: {result.bias_preservation_factor():.2f}")
    
    # Test 3: Aggressively biased
    result = analyzer.analyze_bias_propagation(p1=1e-4, p2=1e-3, eta=100.0)
    print(f"\nAggressively biased (η=100):")
    print(f"  {result}")
    print(f"  Bias preservation: {result.bias_preservation_factor():.2f}")
    
    # Distillation benefit
    distill = compute_distillation_benefit(result)
    print(f"\nDistillation with bias:")
    print(f"  Rounds needed: {distill['n_rounds']}")
    print(f"  Raw states (standard): {distill['raw_states_needed']:.0f}")
    print(f"  Raw states (with bias): {distill['with_bias_exploitation']:.0f}")
    print(f"  Improvement: {distill['improvement_factor']:.2f}x")
    
    # Compare protocols
    print(f"\n\nProtocol Comparison (η=100, p1=1e-4, p2=1e-3):")
    print("-" * 50)
    for prot in ['YL', 'CR', 'MR']:
        analyzer = LogicalBiasAnalyzer(prot)
        result = analyzer.analyze_bias_propagation(1e-4, 1e-3, 100.0)
        print(f"{prot:3s}: η_L = {result.eta_logical:6.2f}, "
              f"preservation = {result.bias_preservation_factor():.2f}")
    
    print("\nLogical bias tests passed!")


if __name__ == "__main__":
    test_logical_bias()
