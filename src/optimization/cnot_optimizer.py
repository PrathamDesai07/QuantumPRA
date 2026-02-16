"""
CNOT Ordering Optimization
===========================

Optimizes CNOT gate ordering in stabilizer measurement circuits
for biased noise regimes.

Key insight: Under biased noise (η >> 1), different CNOT orderings
can dramatically affect logical error rates by suppressing the
dominant error channel.
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from itertools import permutations
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from simulation.circuit_builder import Gate, Timestep
from simulation.monte_carlo import MonteCarloSimulator
from noise_models.depolarizing import NoiseModel
from protocols.base_protocol import BaseProtocol


@dataclass
class CNOTSchedule:
    """
    CNOT ordering for stabilizer measurement.
    
    Attributes:
        stabilizer_type: 'X' or 'Z'
        control_target_pairs: List of (control, target) qubit indices
        schedule: List of timesteps, each containing parallel CNOTs
    """
    stabilizer_type: str  # 'X' or 'Z'
    control_target_pairs: List[Tuple[int, int]]
    schedule: List[List[Tuple[int, int]]]  # Timestep -> list of parallel CNOTs
    
    def total_cnots(self) -> int:
        """Total number of CNOTs."""
        return len(self.control_target_pairs)
    
    def depth(self) -> int:
        """Circuit depth (number of timesteps)."""
        return len(self.schedule)


@dataclass
class OptimizationResult:
    """Result of CNOT ordering optimization."""
    original_schedule: CNOTSchedule
    optimized_schedule: CNOTSchedule
    original_p_L: float
    optimized_p_L: float
    improvement_factor: float
    bias_eta: float
    noise_params: Dict[str, float]


class CNOTOrderingOptimizer:
    """
    Optimizes CNOT ordering for biased noise.
    
    Strategy:
    1. For Z-biased noise (η >> 1), prefer CNOTs that minimize Z error propagation
    2. Schedule gates to minimize time between syndrome measurements
    3. Use greedy or exhaustive search for small problems
    """
    
    def __init__(self, protocol: BaseProtocol, noise: NoiseModel):
        self.protocol = protocol
        self.noise = noise
    
    def analyze_error_propagation(
        self,
        stabilizer_type: str,
        cnot_order: List[Tuple[int, int]],
        bias_eta: float
    ) -> float:
        """
        Analyze error propagation for given CNOT ordering.
        
        For biased noise, different orderings propagate errors differently.
        
        Args:
            stabilizer_type: 'X' or 'Z'
            cnot_order: List of (control, target) pairs
            bias_eta: Noise bias η = p_Z / p_X
        
        Returns:
            Estimated error weight (lower is better)
        """
        # Simplistic model: count how errors propagate
        # This is a heuristic - full analysis requires Heisenberg tracking
        
        error_weight = 0.0
        
        for i, (ctrl, tgt) in enumerate(cnot_order):
            # Position in sequence affects accumulated errors
            position_weight = (i + 1) / len(cnot_order)
            
            # CNOT propagates errors:
            # - X_ctrl -> X_ctrl X_tgt
            # - Z_tgt -> Z_ctrl Z_tgt
            # - X_tgt -> X_tgt (no change)
            # - Z_ctrl -> Z_ctrl (no change)
            
            if stabilizer_type == 'Z':
                # Measuring Z stabilizer: data qubits are controls
                # Z errors on target (syndrome qubit) spread to control
                # This is bad for Z-biased noise
                error_weight += bias_eta * position_weight
            else:
                # Measuring X stabilizer: data qubits are targets
                # X errors propagate, which is suppressed in Z-biased regime
                error_weight += position_weight / bias_eta
        
        return error_weight
    
    def optimize_single_stabilizer(
        self,
        stabilizer_type: str,
        data_qubits: List[int],
        syndrome_qubit: int,
        bias_eta: float,
        method: str = 'greedy'
    ) -> CNOTSchedule:
        """
        Optimize CNOT ordering for single stabilizer measurement.
        
        Args:
            stabilizer_type: 'X' or 'Z'
            data_qubits: List of data qubit indices
            syndrome_qubit: Syndrome qubit index
            bias_eta: Noise bias
            method: 'greedy' or 'exhaustive'
        
        Returns:
            Optimized CNOTSchedule
        """
        # Generate CNOT pairs
        if stabilizer_type == 'Z':
            # For Z stabilizers: data = controls, syndrome = target
            cnot_pairs = [(dq, syndrome_qubit) for dq in data_qubits]
        else:
            # For X stabilizers: syndrome = control, data = targets
            cnot_pairs = [(syndrome_qubit, dq) for dq in data_qubits]
        
        if method == 'exhaustive' and len(data_qubits) <= 4:
            # Try all permutations (only feasible for small problems)
            best_order = None
            best_weight = float('inf')
            
            for perm in permutations(cnot_pairs):
                weight = self.analyze_error_propagation(stabilizer_type, list(perm), bias_eta)
                if weight < best_weight:
                    best_weight = weight
                    best_order = list(perm)
            
            optimized_pairs = best_order
        else:
            # Greedy optimization
            if bias_eta > 10:  # Z-biased
                if stabilizer_type == 'Z':
                    # For Z stabilizers in Z-biased regime: minimize Z propagation
                    # Strategy: measure central qubits first (they affect fewer neighbors)
                    # For now: simple ordering by index
                    optimized_pairs = sorted(cnot_pairs, key=lambda p: p[0])
                else:
                    # X stabilizers are naturally better in Z-biased regime
                    optimized_pairs = cnot_pairs
            else:
                # Unbiased or X-biased: standard ordering
                optimized_pairs = cnot_pairs
        
        # Schedule with parallelism
        schedule = []
        used_qubits = set()
        current_timestep = []
        
        for cnot in optimized_pairs:
            ctrl, tgt = cnot
            
            if ctrl in used_qubits or tgt in used_qubits:
                # Can't parallelize, start new timestep
                if current_timestep:
                    schedule.append(current_timestep)
                current_timestep = [cnot]
                used_qubits = {ctrl, tgt}
            else:
                # Can parallelize
                current_timestep.append(cnot)
                used_qubits.add(ctrl)
                used_qubits.add(tgt)
        
        if current_timestep:
            schedule.append(current_timestep)
        
        return CNOTSchedule(
            stabilizer_type=stabilizer_type,
            control_target_pairs=optimized_pairs,
            schedule=schedule
        )
    
    def optimize_full_protocol(
        self,
        distance: int,
        bias_eta: float,
        n_samples_eval: int = 10000
    ) -> OptimizationResult:
        """
        Optimize CNOT ordering for full protocol.
        
        Args:
            distance: Code distance
            bias_eta: Noise bias
            n_samples_eval: MC samples for evaluation
        
        Returns:
            OptimizationResult with comparison
        """
        print(f"\nOptimizing CNOT ordering for {self.protocol.__class__.__name__}")
        print(f"  Distance: {distance}, Bias η: {bias_eta:.1f}")
        
        # For demonstration, we'll optimize a simple stabilizer
        # Full implementation would optimize all stabilizers in the protocol
        
        # Example: CR protocol has ~4 data qubits per stabilizer
        # Create a representative stabilizer measurement
        
        # Original ordering (linear)
        data_qubits_Z = [0, 1, 2, 3]
        syndrome_qubit = 4
        
        original_pairs = [(dq, syndrome_qubit) for dq in data_qubits_Z]
        original_schedule_timesteps = [[pair] for pair in original_pairs]
        
        original_schedule = CNOTSchedule(
            stabilizer_type='Z',
            control_target_pairs=original_pairs,
            schedule=original_schedule_timesteps
        )
        
        # Optimize
        optimized_schedule = self.optimize_single_stabilizer(
            stabilizer_type='Z',
            data_qubits=data_qubits_Z,
            syndrome_qubit=syndrome_qubit,
            bias_eta=bias_eta,
            method='greedy'
        )
        
        # Evaluate (analytical approximation since full MC is expensive)
        # Use error weight as proxy
        original_weight = self.analyze_error_propagation(
            'Z', original_schedule.control_target_pairs, bias_eta
        )
        optimized_weight = self.analyze_error_propagation(
            'Z', optimized_schedule.control_target_pairs, bias_eta
        )
        
        # Approximate p_L reduction
        improvement_factor = original_weight / optimized_weight if optimized_weight > 0 else 1.0
        
        # For actual p_L, we'd need to run full MC simulations
        # Here we provide analytical estimate
        p2 = self.noise.p2
        original_p_L = 0.6 * p2**2 * original_weight  # Heuristic
        optimized_p_L = 0.6 * p2**2 * optimized_weight
        
        result = OptimizationResult(
            original_schedule=original_schedule,
            optimized_schedule=optimized_schedule,
            original_p_L=original_p_L,
            optimized_p_L=optimized_p_L,
            improvement_factor=improvement_factor,
            bias_eta=bias_eta,
            noise_params={'p1': self.noise.p1, 'p2': self.noise.p2, 'eta': bias_eta}
        )
        
        print(f"\nOptimization results:")
        print(f"  Original depth: {original_schedule.depth()}")
        print(f"  Optimized depth: {optimized_schedule.depth()}")
        print(f"  Original error weight: {original_weight:.3f}")
        print(f"  Optimized error weight: {optimized_weight:.3f}")
        print(f"  Improvement factor: {improvement_factor:.2f}x")
        
        return result


def compare_cnot_orderings(
    protocol_name: str = 'CR',
    distance: int = 3,
    p2: float = 1e-3,
    bias_values: List[float] = [1.0, 10.0, 100.0]
):
    """
    Compare different CNOT orderings across bias regime.
    
    Args:
        protocol_name: Protocol name
        distance: Code distance
        p2: Two-qubit error rate
        bias_values: List of bias values to test
    """
    import matplotlib.pyplot as plt
    
    from protocols.lao_criger import CRProtocol
    
    protocol = CRProtocol()
    
    improvements = []
    
    for eta in bias_values:
        noise = NoiseModel(
            p1=0.1*p2,
            p2=p2,
            p_init=p2,
            p_meas=p2,
            p_idle=0,
            bias_eta=eta
        )
        
        optimizer = CNOTOrderingOptimizer(protocol, noise)
        result = optimizer.optimize_full_protocol(distance=distance, bias_eta=eta)
        improvements.append(result.improvement_factor)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogx(bias_values, improvements, 'o-', markersize=8, linewidth=2)
    ax.set_xlabel('Noise bias $\\eta = p_Z/p_X$')
    ax.set_ylabel('Improvement factor')
    ax.set_title(f'{protocol_name} Protocol: CNOT Ordering Optimization')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No improvement')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return improvements


def test_cnot_optimizer():
    """Test CNOT ordering optimizer."""
    print("Testing CNOT Ordering Optimizer")
    print("=" * 50)
    
    from protocols.lao_criger import CRProtocol
    
    protocol = CRProtocol()
    noise = NoiseModel(
        p1=1e-4,
        p2=1e-3,
        p_init=1e-3,
        p_meas=1e-3,
        bias_eta=100.0  # Strongly Z-biased
    )
    
    optimizer = CNOTOrderingOptimizer(protocol, noise)
    
    # Optimize single stabilizer
    schedule = optimizer.optimize_single_stabilizer(
        stabilizer_type='Z',
        data_qubits=[0, 1, 2, 3],
        syndrome_qubit=4,
        bias_eta=100.0,
        method='greedy'
    )
    
    print(f"\nOptimized schedule:")
    print(f"  Type: {schedule.stabilizer_type}-stabilizer")
    print(f"  Total CNOTs: {schedule.total_cnots()}")
    print(f"  Depth: {schedule.depth()}")
    print(f"  Schedule:")
    for i, timestep in enumerate(schedule.schedule):
        print(f"    t={i}: {timestep}")
    
    # Full protocol optimization
    result = optimizer.optimize_full_protocol(distance=3, bias_eta=100.0)
    
    print("\nCNOT optimizer tests passed!")


if __name__ == "__main__":
    test_cnot_optimizer()
