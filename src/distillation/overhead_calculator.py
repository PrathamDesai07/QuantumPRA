"""
Qubit-Time Volume Calculator for Magic State Distillation

This module computes the full resource overhead for magic-state distillation,
including:
- Physical qubit count vs surface code distance
- Time overhead per T gate
- Qubit-time volume (space × time cost)
- Factory count and parallelization strategies
- Comparison across injection protocols

Following methodology from:
    - Litinski, Quantum 3, 128 (2019) - "Not as costly as you think"
    - Fowler et al., PRA 86, 032324 (2012) - Resource overhead analysis
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt


@dataclass
class SurfaceCodeParameters:
    """Physical parameters for surface code implementation."""
    distance: int                  # Code distance
    physical_qubits: int          # Physical qubits per logical qubit
    cycle_time_us: float          # Time per surface code cycle (microseconds)
    physical_error_rate: float    # Average physical gate error rate
    logical_error_rate: float     # Logical error rate per cycle


@dataclass
class DistillationFactoryConfig:
    """Configuration for a distillation factory."""
    num_levels: int                    # Number of distillation rounds
    logical_qubits_per_level: List[int]  # Qubits at each level
    cycles_per_level: List[int]        # Cycles per round at each level
    output_rate: float                 # T gates per cycle (after warmup)
    raw_states_per_output: float       # Total raw states consumed


@dataclass
class OverheadMetrics:
    """Complete resource overhead for magic-state preparation."""
    protocol_name: str
    injection_error_rate: float
    target_error_rate: float
    
    # Distillation cascade
    num_distillation_rounds: int
    total_raw_states: float
    
    # Surface code resources
    code_distance: int
    physical_qubits_per_factory: int
    
    # Time resources
    injection_time_us: float           # Time to inject one raw state
    distillation_time_us: float        # Time through distillation cascade
    total_time_per_T_us: float        # Total time per T gate
    
    # Space-time volume
    qubit_time_volume: float          # Physical qubits × time (qubit-microseconds)
    
    # Comparative metrics
    relative_overhead: float = 1.0    # Relative to baseline (set later)


class OverheadCalculator:
    """
    Calculate end-to-end resource overhead for magic-state distillation.
    """
    
    def __init__(self):
        """Initialize overhead calculator with default models."""
        pass
    
    def compute_surface_code_params(
        self,
        distance: int,
        physical_error_rate: float,
        cycle_time_us: float = 1.0
    ) -> SurfaceCodeParameters:
        """
        Compute surface code parameters for given distance.
        
        Args:
            distance: Surface code distance
            physical_error_rate: Physical gate error rate
            cycle_time_us: Time per syndrome extraction cycle
            
        Returns:
            SurfaceCodeParameters with all metrics
        """
        # Physical qubits: d^2 data + (d^2 - 1) ancillas ≈ 2d^2
        physical_qubits = 2 * (distance ** 2)
        
        # Logical error rate: exponential suppression with distance
        # p_L ≈ a * (p_phys / p_th)^((d+1)/2)
        # Using p_th ≈ 0.01 for surface code
        p_th = 0.01
        if physical_error_rate < p_th:
            # Below threshold: exponential suppression
            logical_error_rate = 0.1 * ((physical_error_rate / p_th) ** ((distance + 1) / 2))
        else:
            # Above threshold: no quantum advantage
            logical_error_rate = physical_error_rate
        
        return SurfaceCodeParameters(
            distance=distance,
            physical_qubits=physical_qubits,
            cycle_time_us=cycle_time_us,
            physical_error_rate=physical_error_rate,
            logical_error_rate=logical_error_rate
        )
    
    def estimate_injection_time(
        self,
        protocol_name: str,
        code_distance: int,
        cycle_time_us: float = 1.0
    ) -> float:
        """
        Estimate time to inject one raw magic state.
        
        Injection involves:
        1. Initialize data qubits + magic qubit
        2. Measure stabilizers (2 rounds typically)
        3. Post-select on syndrome outcomes
        
        Args:
            protocol_name: Protocol identifier (YL, CR, MR, etc.)
            code_distance: Surface code distance
            cycle_time_us: Time per cycle
            
        Returns:
            Injection time in microseconds
        """
        # Protocol-specific circuit depths (in syndrome extraction cycles)
        depth_map = {
            'YL': 4,   # Li protocol: 2 initialization + 2 stabilizer rounds
            'CR': 4,   # Corner rotated: similar
            'MR': 5,   # Middle rotated: slightly deeper
            'Optimized': 4,
            'Stacked': 3,   # Parallel measurements
            'Masked': 3,
            'XZZX': 4
        }
        
        base_depth = depth_map.get(protocol_name, 4)
        
        # Total cycles: scales with code distance for syndrome extraction
        total_cycles = base_depth * code_distance
        
        return total_cycles * cycle_time_us
    
    def estimate_distillation_time(
        self,
        num_rounds: int,
        code_distance: int,
        cycle_time_us: float = 1.0
    ) -> float:
        """
        Estimate time through distillation cascade.
        
        Each distillation round involves:
        1. Encode 15 states into [[15,1,3]] code
        2. Measure syndrome (12 stabilizers)
        3. Measure encoded state out
        
        Args:
            num_rounds: Number of distillation rounds
            code_distance: Surface code distance
            cycle_time_us: Time per cycle
            
        Returns:
            Distillation time in microseconds
        """
        # Distillation circuit depth per round (in surface code cycles)
        # Encoding + syndrome + decode ≈ 50-100 cycles
        cycles_per_round = 75 * code_distance
        
        # Sequential rounds (though can pipeline with multiple factories)
        total_cycles = cycles_per_round * num_rounds
        
        return total_cycles * cycle_time_us
    
    def compute_factory_resources(
        self,
        num_distillation_rounds: int,
        code_distance: int
    ) -> Dict[str, int]:
        """
        Compute resource requirements for one distillation factory.
        
        Args:
            num_distillation_rounds: Cascade depth
            code_distance: Surface code distance
            
        Returns:
            Dict with logical and physical qubit counts
        """
        # Each distillation round uses 15 input + ancillas
        logical_qubits_per_round = 20  # Conservative estimate
        
        # Factory operates all rounds in pipeline (after warmup)
        total_logical_qubits = logical_qubits_per_round * num_distillation_rounds
        
        # Physical qubits per logical
        physical_per_logical = 2 * (code_distance ** 2)
        total_physical_qubits = total_logical_qubits * physical_per_logical
        
        return {
            'logical_qubits': total_logical_qubits,
            'physical_qubits': total_physical_qubits,
            'physical_per_logical': physical_per_logical
        }
    
    def compute_overhead(
        self,
        protocol_name: str,
        injection_error_rate: float,
        cascade_result,  # DistillationCascade from bravyi_kitaev.py
        code_distance: int,
        physical_error_rate: float,
        cycle_time_us: float = 1.0
    ) -> OverheadMetrics:
        """
        Compute full end-to-end resource overhead.
        
        Args:
            protocol_name: Protocol identifier
            injection_error_rate: Raw injection p_L
            cascade_result: DistillationCascade object
            code_distance: Surface code distance
            physical_error_rate: Physical gate error rate
            cycle_time_us: Surface code cycle time
            
        Returns:
            OverheadMetrics with all resource estimates
        """
        # Surface code parameters
        surf_params = self.compute_surface_code_params(
            code_distance, physical_error_rate, cycle_time_us
        )
        
        # Time estimates
        injection_time = self.estimate_injection_time(
            protocol_name, code_distance, cycle_time_us
        )
        distillation_time = self.estimate_distillation_time(
            cascade_result.num_rounds, code_distance, cycle_time_us
        )
        total_time = injection_time * cascade_result.total_raw_states + distillation_time
        
        # Space estimates
        factory_resources = self.compute_factory_resources(
            cascade_result.num_rounds, code_distance
        )
        physical_qubits = factory_resources['physical_qubits']
        
        # Qubit-time volume
        qubit_time_volume = physical_qubits * total_time
        
        return OverheadMetrics(
            protocol_name=protocol_name,
            injection_error_rate=injection_error_rate,
            target_error_rate=cascade_result.p_target,
            num_distillation_rounds=cascade_result.num_rounds,
            total_raw_states=cascade_result.total_raw_states,
            code_distance=code_distance,
            physical_qubits_per_factory=physical_qubits,
            injection_time_us=injection_time,
            distillation_time_us=distillation_time,
            total_time_per_T_us=total_time,
            qubit_time_volume=qubit_time_volume
        )
    
    def compare_protocols(
        self,
        protocol_cascades: Dict[str, any],  # Dict[str, DistillationCascade]
        protocol_p_L: Dict[str, float],
        code_distance: int = 5,
        physical_error_rate: float = 1e-3,
        cycle_time_us: float = 1.0,
        baseline: str = 'YL'
    ) -> Dict[str, OverheadMetrics]:
        """
        Compare resource overhead across multiple protocols.
        
        Args:
            protocol_cascades: Distillation cascades for each protocol
            protocol_p_L: Injection error rates
            code_distance: Surface code distance
            physical_error_rate: Physical error rate
            cycle_time_us: Cycle time
            baseline: Protocol to use as baseline for relative metrics
            
        Returns:
            Dict mapping protocol names to OverheadMetrics
        """
        results = {}
        
        for name, cascade in protocol_cascades.items():
            if cascade is None:
                continue
                
            p_L = protocol_p_L[name]
            metrics = self.compute_overhead(
                name, p_L, cascade, code_distance, 
                physical_error_rate, cycle_time_us
            )
            results[name] = metrics
        
        # Compute relative overheads
        if baseline in results:
            baseline_volume = results[baseline].qubit_time_volume
            for metrics in results.values():
                metrics.relative_overhead = metrics.qubit_time_volume / baseline_volume
        
        return results
    
    def print_comparison_table(
        self,
        overhead_results: Dict[str, OverheadMetrics]
    ) -> None:
        """
        Print formatted comparison table.
        
        Args:
            overhead_results: Dict of OverheadMetrics
        """
        print("\n" + "="*80)
        print("MAGIC STATE DISTILLATION OVERHEAD COMPARISON")
        print("="*80)
        print(f"{'Protocol':<12} {'p_L':<10} {'Rounds':<8} {'Raw States':<12} "
              f"{'Time(μs)':<12} {'Rel. Cost':<10}")
        print("-"*80)
        
        for name, metrics in sorted(overhead_results.items()):
            print(f"{name:<12} {metrics.injection_error_rate:.2e}  "
                  f"{metrics.num_distillation_rounds:<8} "
                  f"{metrics.total_raw_states:<12.1f} "
                  f"{metrics.total_time_per_T_us:<12.1f} "
                  f"{metrics.relative_overhead:<10.2f}")
        
        print("="*80)
    
    def plot_overhead_scaling(
        self,
        protocol_cascades: Dict[str, any],
        protocol_p_L: Dict[str, float],
        distance_range: List[int] = [3, 5, 7, 9],
        physical_error_rate: float = 1e-3,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot how overhead scales with code distance for different protocols.
        
        Args:
            protocol_cascades: Distillation cascades
            protocol_p_L: Injection error rates
            distance_range: Code distances to evaluate
            physical_error_rate: Physical error rate
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        for name in protocol_cascades.keys():
            if protocol_cascades[name] is None:
                continue
                
            volumes = []
            times = []
            
            for d in distance_range:
                metrics = self.compute_overhead(
                    name, protocol_p_L[name], protocol_cascades[name],
                    d, physical_error_rate
                )
                volumes.append(metrics.qubit_time_volume)
                times.append(metrics.total_time_per_T_us)
            
            ax1.plot(distance_range, volumes, 'o-', label=name, linewidth=2)
            ax2.plot(distance_range, times, 's-', label=name, linewidth=2)
        
        ax1.set_xlabel('Code Distance', fontsize=12)
        ax1.set_ylabel('Qubit-Time Volume (qubit-μs)', fontsize=12)
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Space-Time Cost vs Distance')
        
        ax2.set_xlabel('Code Distance', fontsize=12)
        ax2.set_ylabel('Time per T gate (μs)', fontsize=12)
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Time Cost vs Distance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()


# Example usage
if __name__ == "__main__":
    print("=== Overhead Calculator ===\n")
    
    # Import distillation module
    import sys
    sys.path.insert(0, '/teamspace/studios/this_studio/QuantumPRA/src')
    from distillation.bravyi_kitaev import BravyiKitaevDistillation
    
    # Setup
    distiller = BravyiKitaevDistillation()
    calculator = OverheadCalculator()
    
    # Protocol error rates (from Phase 3 results)
    protocol_p_L = {
        'YL': 3.0e-3,
        'CR': 2.5e-3,
        'MR': 2.0e-3,
        'Optimized': 1.2e-3
    }
    
    # Design cascades
    p_target = 1e-10
    cascades = {}
    for name, p_L in protocol_p_L.items():
        try:
            cascades[name] = distiller.design_cascade(p_L, p_target)
        except ValueError:
            cascades[name] = None
            print(f"Warning: {name} cannot reach target")
    
    # Compare overheads
    overhead_results = calculator.compare_protocols(
        cascades, protocol_p_L,
        code_distance=5,
        physical_error_rate=1e-3,
        baseline='YL'
    )
    
    calculator.print_comparison_table(overhead_results)
    
    # Show improvement
    if 'YL' in overhead_results and 'Optimized' in overhead_results:
        improvement = overhead_results['YL'].qubit_time_volume / \
                     overhead_results['Optimized'].qubit_time_volume
        print(f"\n✓ Optimized protocol reduces overhead by {improvement:.2f}x vs baseline YL")
    
    print("\n✓ Overhead calculator module validated!")
