"""
Parameter Space Sweeping
=========================

Systematically explores parameter space for magic-state injection protocols.
Scans over code distance, noise parameters, and bias to characterize protocol performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from itertools import product
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.append(str(Path(__file__).parent.parent))

from simulation.monte_carlo import MonteCarloSimulator, SimulationResult
from noise_models.depolarizing import NoiseModel
from protocols.base_protocol import BaseProtocol
from protocols.lao_criger import CRProtocol, MRProtocol
from protocols.li_protocol import LiProtocol
from analysis.symbolic_expressions import SymbolicAnalyzer


@dataclass
class SweepConfiguration:
    """
    Configuration for parameter sweep.
    
    Attributes:
        protocols: List of protocol names to compare
        distances: List of code distances
        p2_values: Two-qubit error rates to scan
        p1_ratio: Ratio p1/p2 (default 0.1)
        bias_values: Noise bias values η = p_Z/p_X
        n_samples: Monte Carlo samples per point
        n_workers: Parallel workers
    """
    protocols: List[str]
    distances: List[int]
    p2_values: List[float]
    p1_ratio: float = 0.1
    bias_values: List[float] = None
    n_samples: int = 100000
    n_workers: int = 4
    
    def __post_init__(self):
        if self.bias_values is None:
            self.bias_values = [1.0]  # Unbiased by default
    
    def total_points(self) -> int:
        """Total number of parameter points."""
        return len(self.protocols) * len(self.distances) * len(self.p2_values) * len(self.bias_values)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SweepPoint:
    """Single point in parameter sweep."""
    protocol_name: str
    distance: int
    p1: float
    p2: float
    bias_eta: float
    result: SimulationResult = None
    

class ParameterSweeper:
    """
    Systematically sweeps parameter space for protocol comparison.
    """
    
    def __init__(self, config: SweepConfiguration, output_dir: str = "data/sweeps"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = SymbolicAnalyzer()
        self.results: List[SweepPoint] = []
    
    def _get_protocol(self, name: str, distance: int = 3) -> BaseProtocol:
        """Get protocol instance by name."""
        if name == 'YL':
            return LiProtocol(distance=distance)
        elif name == 'CR':
            return CRProtocol(distance=distance)
        elif name == 'MR':
            return MRProtocol(distance=distance)
        else:
            raise ValueError(f"Unknown protocol: {name}")
    
    def _run_single_point(self, point: SweepPoint) -> SweepPoint:
        """
        Run simulation for single parameter point.
        
        Args:
            point: SweepPoint with parameters
        
        Returns:
            SweepPoint with result populated
        """
        # Create noise model
        noise = NoiseModel(
            p1=point.p1,
            p2=point.p2,
            p_init=point.p2,
            p_meas=point.p2,
            p_idle=0,
            bias_eta=point.bias_eta
        )
        
        # Get protocol with distance
        protocol = self._get_protocol(point.protocol_name, point.distance)
        
        # Run simulation
        simulator = MonteCarloSimulator(protocol, noise)
        result = simulator.run(
            n_samples=self.config.n_samples
        )
        
        point.result = result
        return point
    
    def run_sweep(self, parallel: bool = True) -> List[SweepPoint]:
        """
        Run full parameter sweep.
        
        Args:
            parallel: Use parallel execution
        
        Returns:
            List of SweepPoints with results
        """
        # Generate all parameter combinations
        points = []
        for prot, dist, p2, eta in product(
            self.config.protocols,
            self.config.distances,
            self.config.p2_values,
            self.config.bias_values
        ):
            p1 = self.config.p1_ratio * p2
            point = SweepPoint(
                protocol_name=prot,
                distance=dist,
                p1=p1,
                p2=p2,
                bias_eta=eta
            )
            points.append(point)
        
        print(f"Running parameter sweep: {len(points)} points")
        print(f"  Protocols: {self.config.protocols}")
        print(f"  Distances: {self.config.distances}")
        print(f"  p2 range: {min(self.config.p2_values):.1e} - {max(self.config.p2_values):.1e}")
        print(f"  Bias values: {self.config.bias_values}")
        print(f"  Samples per point: {self.config.n_samples}")
        
        start_time = time.time()
        
        if parallel and self.config.n_workers > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
                futures = {executor.submit(self._run_single_point, p): p for p in points}
                
                completed = 0
                for future in as_completed(futures):
                    point = future.result()
                    self.results.append(point)
                    completed += 1
                    
                    if completed % 10 == 0 or completed == len(points):
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        remaining = (len(points) - completed) / rate if rate > 0 else 0
                        print(f"  Progress: {completed}/{len(points)} ({100*completed/len(points):.1f}%), "
                              f"ETA: {remaining/60:.1f} min")
        else:
            # Serial execution
            for i, point in enumerate(points):
                result_point = self._run_single_point(point)
                self.results.append(result_point)
                
                if (i + 1) % 5 == 0 or i == len(points) - 1:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (len(points) - (i + 1)) / rate if rate > 0 else 0
                    print(f"  Progress: {i+1}/{len(points)} ({100*(i+1)/len(points):.1f}%), "
                          f"ETA: {remaining/60:.1f} min")
        
        total_time = time.time() - start_time
        print(f"\nSweep completed in {total_time/60:.1f} minutes")
        
        return self.results
    
    def save_results(self, filename: str = None):
        """Save sweep results to JSON."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"sweep_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert to serializable format
        data = {
            'config': self.config.to_dict(),
            'results': []
        }
        
        for point in self.results:
            point_data = {
                'protocol_name': point.protocol_name,
                'distance': point.distance,
                'p1': point.p1,
                'p2': point.p2,
                'bias_eta': point.bias_eta,
                'result': asdict(point.result) if point.result else None
            }
            data['results'].append(point_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filepath}")
        return filepath
    
    def plot_sweep_results(self, save_path: str = None):
        """
        Plot sweep results: p_L vs p2 for different protocols.
        
        Args:
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, len(self.config.distances), 
                                figsize=(6*len(self.config.distances), 5))
        
        if len(self.config.distances) == 1:
            axes = [axes]
        
        # Group by distance
        for ax, distance in zip(axes, self.config.distances):
            # Plot each protocol
            for protocol in self.config.protocols:
                # Get points for this protocol and distance
                points = [p for p in self.results 
                         if p.protocol_name == protocol and p.distance == distance]
                
                if not points:
                    continue
                
                # Group by bias
                for eta in self.config.bias_values:
                    eta_points = [p for p in points if p.bias_eta == eta]
                    eta_points.sort(key=lambda p: p.p2)
                    
                    p2_vals = [p.p2 for p in eta_points]
                    p_L_vals = [p.result.p_logical for p in eta_points if p.result]
                    
                    label = f"{protocol} (η={eta:.0f})" if len(self.config.bias_values) > 1 else protocol
                    ax.loglog(p2_vals, p_L_vals, 'o-', label=label, markersize=6)
                    
                    # Add analytic curve if η = 1 (unbiased)
                    if eta == 1.0:
                        if protocol == 'CR':
                            expr = self.analyzer.derive_CR_expression()
                        elif protocol == 'MR':
                            expr = self.analyzer.derive_MR_expression()
                        elif protocol == 'YL':
                            expr = self.analyzer.derive_YL_expression()
                        else:
                            continue
                        
                        p2_range = np.logspace(np.log10(min(p2_vals)), np.log10(max(p2_vals)), 50)
                        p_L_analytic = []
                        for p2 in p2_range:
                            p1 = self.config.p1_ratio * p2
                            p_L = expr.evaluate(p_1=p1, p_2=p2, p_I=p2, p_M=p2)
                            p_L_analytic.append(p_L)
                        
                        ax.loglog(p2_range, p_L_analytic, '--', alpha=0.5, 
                                label=f'{protocol} (analytic)')
            
            ax.set_xlabel('Two-qubit error rate $p_2$')
            ax.set_ylabel('Logical error rate $p_L$')
            ax.set_title(f'Distance {distance}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        return fig, axes
    
    def generate_comparison_table(self) -> str:
        """
        Generate comparison table of protocols.
        
        Returns:
            Formatted table string
        """
        # Group by distance and p2
        table = "Protocol Comparison\n"
        table += "=" * 80 + "\n\n"
        
        for distance in self.config.distances:
            table += f"Distance {distance}:\n"
            table += "-" * 80 + "\n"
            
            # Header
            table += f"{'p2':>10s} {'η':>8s} "
            for prot in self.config.protocols:
                table += f"{prot:>12s} "
            table += "\n"
            
            # Group by p2 and bias
            for p2 in sorted(self.config.p2_values):
                for eta in self.config.bias_values:
                    table += f"{p2:>10.1e} {eta:>8.1f} "
                    
                    for prot in self.config.protocols:
                        points = [p for p in self.results 
                                 if p.protocol_name == prot and p.distance == distance 
                                 and p.p2 == p2 and p.bias_eta == eta]
                        
                        if points and points[0].result:
                            p_L = points[0].result.p_logical
                            table += f"{p_L:>12.4e} "
                        else:
                            table += f"{'N/A':>12s} "
                    
                    table += "\n"
            
            table += "\n"
        
        return table


def quick_sweep_unbiased(protocols: List[str] = ['CR', 'MR'], 
                         distance: int = 3,
                         n_samples: int = 10000) -> ParameterSweeper:
    """
    Quick parameter sweep for unbiased noise.
    
    Args:
        protocols: List of protocols to compare
        distance: Code distance
        n_samples: Samples per point
    
    Returns:
        Completed ParameterSweeper
    """
    config = SweepConfiguration(
        protocols=protocols,
        distances=[distance],
        p2_values=[1e-4, 3e-4, 1e-3, 3e-3],
        p1_ratio=0.1,
        bias_values=[1.0],
        n_samples=n_samples,
        n_workers=4
    )
    
    sweeper = ParameterSweeper(config)
    sweeper.run_sweep(parallel=True)
    sweeper.save_results()
    sweeper.plot_sweep_results()
    
    return sweeper


def bias_dependence_sweep(protocol: str = 'MR',
                          distance: int = 3,
                          p2: float = 1e-3,
                          n_samples: int = 10000) -> ParameterSweeper:
    """
    Sweep bias parameter for single protocol.
    
    Args:
        protocol: Protocol name
        distance: Code distance
        p2: Two-qubit error rate
        n_samples: Samples per point
    
    Returns:
        Completed ParameterSweeper
    """
    config = SweepConfiguration(
        protocols=[protocol],
        distances=[distance],
        p2_values=[p2],
        p1_ratio=0.1,
        bias_values=[1.0, 3.0, 10.0, 30.0, 100.0],
        n_samples=n_samples,
        n_workers=5
    )
    
    sweeper = ParameterSweeper(config)
    sweeper.run_sweep(parallel=True)
    sweeper.save_results()
    
    # Custom plot for bias dependence
    fig, ax = plt.subplots(figsize=(8, 6))
    
    eta_vals = []
    p_L_vals = []
    eta_L_vals = []
    
    for point in sweeper.results:
        if point.result:
            eta_vals.append(point.bias_eta)
            p_L_vals.append(point.result.p_logical)
            eta_L_vals.append(point.result.logical_bias)
    
    ax.loglog(eta_vals, p_L_vals, 'o-', label='$p_L$', markersize=8)
    ax.set_xlabel('Physical bias $\\eta = p_Z/p_X$')
    ax.set_ylabel('Error rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{protocol} Protocol: Bias Dependence (d={distance}, $p_2$={p2:.1e})')
    
    plt.tight_layout()
    plt.show()
    
    return sweeper


if __name__ == "__main__":
    print("Parameter Space Sweeping Demo")
    print("=" * 50)
    
    # Quick demo with small sample size
    print("\nRunning quick sweep (small sample size for demo)...")
    sweeper = quick_sweep_unbiased(protocols=['CR', 'MR'], distance=3, n_samples=1000)
    
    print("\nComparison table:")
    print(sweeper.generate_comparison_table())
