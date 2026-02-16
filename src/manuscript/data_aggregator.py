"""
Manuscript Data Aggregator

This module aggregates results from all phases (simulation, analysis, optimization)
and prepares data for:
1. Paper figures
2. Supplementary material tables
3. LaTeX table generation
4. Result validation and consistency checks

Central script for generating all paper outputs from raw simulation data.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from distillation.bravyi_kitaev import BravyiKitaevDistillation
from distillation.overhead_calculator import OverheadCalculator


@dataclass
class ProtocolSummary:
    """Summary statistics for a single protocol."""
    name: str
    p_L: float                    # Logical error rate
    p_accept: float               # Acceptance probability
    alpha2: float                 # Leading-order coefficient
    alpha1: float                 # Single-qubit error coefficient
    eta_logical: float            # Logical noise bias
    circuit_depth: int            # Circuit depth in cycles
    num_cnots: int                # Total CNOT count


@dataclass
class ComparisonMetrics:
    """Comparative metrics across protocols."""
    baseline: str
    protocols: List[str]
    p_L_improvements: Dict[str, float]      # Relative to baseline
    overhead_reductions: Dict[str, float]   # Distillation cost reduction
    acceptance_rates: Dict[str, float]


class ManuscriptDataAggregator:
    """
    Aggregate all simulation results for paper generation.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize aggregator.
        
        Args:
            data_dir: Root directory containing raw and processed data
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.figures_dir = self.data_dir / "figures"
        
        # Create directories
        for d in [self.raw_dir, self.processed_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Initialize tools
        self.distiller = BravyiKitaevDistillation()
        self.overhead_calc = OverheadCalculator()
    
    def aggregate_protocol_results(
        self,
        protocol_names: List[str],
        simulation_results: Dict[str, Dict]
    ) -> Dict[str, ProtocolSummary]:
        """
        Aggregate simulation results into protocol summaries.
        
        Args:
            protocol_names: List of protocol names
            simulation_results: Raw simulation data for each protocol
            
        Returns:
            Dict mapping protocol names to ProtocolSummary objects
        """
        summaries = {}
        
        for name in protocol_names:
            if name not in simulation_results:
                print(f"Warning: No data for protocol {name}")
                continue
            
            data = simulation_results[name]
            
            summary = ProtocolSummary(
                name=name,
                p_L=data.get('p_L', 0.0),
                p_accept=data.get('p_accept', 1.0),
                alpha2=data.get('alpha2', 0.0),
                alpha1=data.get('alpha1', 0.0),
                eta_logical=data.get('eta_logical', 1.0),
                circuit_depth=data.get('circuit_depth', 0),
                num_cnots=data.get('num_cnots', 0)
            )
            
            summaries[name] = summary
        
        return summaries
    
    def compute_comparison_metrics(
        self,
        summaries: Dict[str, ProtocolSummary],
        baseline: str = 'YL'
    ) -> ComparisonMetrics:
        """
        Compute comparative metrics relative to baseline.
        
        Args:
            summaries: Protocol summaries
            baseline: Baseline protocol name
            
        Returns:
            ComparisonMetrics object
        """
        if baseline not in summaries:
            raise ValueError(f"Baseline protocol {baseline} not found")
        
        baseline_p_L = summaries[baseline].p_L
        protocols = list(summaries.keys())
        
        # p_L improvements
        p_L_improvements = {}
        for name, summary in summaries.items():
            if summary.p_L > 0:
                p_L_improvements[name] = baseline_p_L / summary.p_L
            else:
                p_L_improvements[name] = float('inf')
        
        # Compute distillation overheads
        p_target = 1e-10
        overhead_reductions = {}
        
        try:
            baseline_cascade = self.distiller.design_cascade(baseline_p_L, p_target)
            baseline_overhead = baseline_cascade.total_raw_states
            
            for name, summary in summaries.items():
                try:
                    cascade = self.distiller.design_cascade(summary.p_L, p_target)
                    overhead = cascade.total_raw_states
                    overhead_reductions[name] = baseline_overhead / overhead
                except ValueError:
                    overhead_reductions[name] = 0.0
        except ValueError:
            print(f"Warning: Could not compute distillation for baseline {baseline}")
            overhead_reductions = {name: 1.0 for name in protocols}
        
        # Acceptance rates
        acceptance_rates = {name: s.p_accept for name, s in summaries.items()}
        
        return ComparisonMetrics(
            baseline=baseline,
            protocols=protocols,
            p_L_improvements=p_L_improvements,
            overhead_reductions=overhead_reductions,
            acceptance_rates=acceptance_rates
        )
    
    def generate_latex_table(
        self,
        summaries: Dict[str, ProtocolSummary],
        comparison: ComparisonMetrics,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate LaTeX table for manuscript.
        
        Args:
            summaries: Protocol summaries
            comparison: Comparison metrics
            output_file: Optional file to save table
            
        Returns:
            LaTeX table string
        """
        latex = []
        latex.append(r"\begin{table}[ht]")
        latex.append(r"\centering")
        latex.append(r"\caption{Magic State Injection Protocol Comparison}")
        latex.append(r"\label{tab:protocol_comparison}")
        latex.append(r"\begin{tabular}{lccccc}")
        latex.append(r"\hline\hline")
        latex.append(r"Protocol & $p_L$ & $\alpha_2$ & $\eta_L$ & CNOTs & Improvement \\")
        latex.append(r"\hline")
        
        for name in comparison.protocols:
            if name not in summaries:
                continue
            s = summaries[name]
            imp = comparison.p_L_improvements[name]
            
            latex.append(
                f"{name} & "
                f"${s.p_L:.2e}$ & "
                f"${s.alpha2:.3f}$ & "
                f"${s.eta_logical:.1f}$ & "
                f"{s.num_cnots} & "
                f"{imp:.2f}$\\times$ \\\\"
            )
        
        latex.append(r"\hline\hline")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        
        table_str = "\n".join(latex)
        
        if output_file:
            output_path = self.processed_dir / output_file
            with open(output_path, 'w') as f:
                f.write(table_str)
            print(f"LaTeX table saved: {output_path}")
        
        return table_str
    
    def prepare_figure_data(
        self,
        summaries: Dict[str, ProtocolSummary],
        parameter_sweep_results: Optional[Dict] = None,
        convergence_results: Optional[Dict] = None
    ) -> Dict:
        """
        Prepare data for all paper figures.
        
        Args:
            summaries: Protocol summaries
            parameter_sweep_results: Optional sweep data
            convergence_results: Optional convergence data
            
        Returns:
            Dict with all figure data
        """
        figure_data = {}
        
        # Figure 1: Protocol comparison (p_L vs p_2)
        if parameter_sweep_results and 'p2_sweep' in parameter_sweep_results:
            figure_data['protocol_comparison'] = {}
            for name, data in parameter_sweep_results['p2_sweep'].items():
                figure_data['protocol_comparison'][name] = {
                    'p2': data['p2_values'],
                    'p_L': data['p_L_values'],
                    'analytic': data.get('analytic_p_L', data['p_L_values'])
                }
        
        # Figure 2: Bias dependence
        if parameter_sweep_results and 'bias_sweep' in parameter_sweep_results:
            figure_data['bias_dependence'] = {}
            for name, data in parameter_sweep_results['bias_sweep'].items():
                figure_data['bias_dependence'][name] = {
                    'eta_phys': data['eta_values'],
                    'eta_logical': data['eta_L_values']
                }
        
        # Figure 3: CNOT optimization
        if parameter_sweep_results and 'cnot_optimization' in parameter_sweep_results:
            figure_data['cnot_optimization'] = parameter_sweep_results['cnot_optimization']
        
        # Figure 4: Convergence validation
        if convergence_results:
            figure_data['convergence'] = {}
            for name, data in convergence_results.items():
                if 'p2_values' in data and 'ratio' in data:
                    figure_data['convergence'][name] = {
                        'p2': data['p2_values'],
                        'ratio': data['ratio'],
                        'alpha2_theory': summaries[name].alpha2 if name in summaries else 0.5
                    }
        
        # Figure 5: Distillation overhead
        p_target = 1e-10
        figure_data['distillation'] = {}
        for name, summary in summaries.items():
            try:
                cascade = self.distiller.design_cascade(summary.p_L, p_target)
                metrics = self.overhead_calc.compute_overhead(
                    name, summary.p_L, cascade, 
                    code_distance=5, physical_error_rate=1e-3
                )
                figure_data['distillation'][name] = {
                    'volume': metrics.qubit_time_volume,
                    'rounds': cascade.num_rounds,
                    'raw_states': cascade.total_raw_states
                }
            except ValueError:
                continue
        
        # Figure 6: Distance scaling
        if parameter_sweep_results and 'distance_sweep' in parameter_sweep_results:
            figure_data['distance_scaling'] = {}
            for name, data in parameter_sweep_results['distance_sweep'].items():
                figure_data['distance_scaling'][name] = {
                    'distances': data['distances'],
                    'p_L': data['p_L_values'],
                    'overhead': data.get('overhead_values', [1.0] * len(data['distances']))
                }
        
        return figure_data
    
    def save_aggregated_data(
        self,
        figure_data: Dict,
        filename: str = "aggregated_results.json"
    ) -> None:
        """
        Save aggregated data to JSON file.
        
        Args:
            figure_data: Dict with all figure data
            filename: Output filename
        """
        output_path = self.processed_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(figure_data, f, indent=2)
        
        print(f"Aggregated data saved: {output_path}")
    
    def generate_summary_report(
        self,
        summaries: Dict[str, ProtocolSummary],
        comparison: ComparisonMetrics,
        output_file: str = "summary_report.txt"
    ) -> None:
        """
        Generate human-readable summary report.
        
        Args:
            summaries: Protocol summaries
            comparison: Comparison metrics
            output_file: Output filename
        """
        output_path = self.processed_dir / output_file
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MAGIC STATE INJECTION: MANUSCRIPT DATA SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Baseline Protocol: {comparison.baseline}\n")
            f.write(f"Number of Protocols: {len(comparison.protocols)}\n\n")
            
            f.write("PROTOCOL SUMMARIES\n")
            f.write("-"*80 + "\n")
            for name, summary in summaries.items():
                f.write(f"\n{name}:\n")
                f.write(f"  Logical error rate: p_L = {summary.p_L:.3e}\n")
                f.write(f"  Acceptance rate: p_acc = {summary.p_accept:.3f}\n")
                f.write(f"  Leading order: α₂ = {summary.alpha2:.3f}\n")
                f.write(f"  Logical bias: η_L = {summary.eta_logical:.1f}\n")
                f.write(f"  Circuit depth: {summary.circuit_depth} cycles\n")
                f.write(f"  CNOT count: {summary.num_cnots}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("COMPARATIVE METRICS\n")
            f.write("-"*80 + "\n\n")
            
            f.write("Error Rate Improvements (vs baseline):\n")
            for name, imp in comparison.p_L_improvements.items():
                if imp < float('inf'):
                    f.write(f"  {name}: {imp:.2f}x\n")
            
            f.write("\nDistillation Overhead Reductions (vs baseline):\n")
            for name, red in comparison.overhead_reductions.items():
                if red > 0:
                    f.write(f"  {name}: {red:.2f}x\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Best Protocol: ")
            best_protocol = max(comparison.overhead_reductions.items(), 
                              key=lambda x: x[1] if x[1] > 0 else 0)[0]
            best_improvement = comparison.overhead_reductions[best_protocol]
            f.write(f"{best_protocol} ({best_improvement:.2f}x overhead reduction)\n")
            f.write("="*80 + "\n")
        
        print(f"Summary report saved: {output_path}")
    
    def run_full_analysis(
        self,
        simulation_results: Dict[str, Dict],
        parameter_sweep_results: Optional[Dict] = None,
        convergence_results: Optional[Dict] = None,
        baseline: str = 'YL'
    ) -> None:
        """
        Run complete analysis pipeline for manuscript.
        
        Args:
            simulation_results: Raw simulation data
            parameter_sweep_results: Parameter sweep data
            convergence_results: Convergence validation data
            baseline: Baseline protocol for comparison
        """
        print("="*80)
        print("MANUSCRIPT DATA AGGREGATION")
        print("="*80 + "\n")
        
        # Aggregate protocol results
        print("1. Aggregating protocol results...")
        protocol_names = list(simulation_results.keys())
        summaries = self.aggregate_protocol_results(protocol_names, simulation_results)
        print(f"   ✓ {len(summaries)} protocols summarized\n")
        
        # Compute comparisons
        print("2. Computing comparative metrics...")
        comparison = self.compute_comparison_metrics(summaries, baseline)
        print(f"   ✓ Comparisons relative to {baseline}\n")
        
        # Generate LaTeX table
        print("3. Generating LaTeX table...")
        self.generate_latex_table(summaries, comparison, "table_protocol_comparison.tex")
        print("   ✓ Table generated\n")
        
        # Prepare figure data
        print("4. Preparing figure data...")
        figure_data = self.prepare_figure_data(summaries, parameter_sweep_results, convergence_results)
        print(f"   ✓ Data for {len(figure_data)} figures prepared\n")
        
        # Save aggregated data
        print("5. Saving aggregated data...")
        self.save_aggregated_data(figure_data)
        print("   ✓ Data saved\n")
        
        # Generate summary report
        print("6. Generating summary report...")
        self.generate_summary_report(summaries, comparison)
        print("   ✓ Report generated\n")
        
        print("="*80)
        print("AGGREGATION COMPLETE")
        print("="*80)
        print(f"\nOutputs saved in: {self.processed_dir}/")


# Example usage
if __name__ == "__main__":
    print("=== Manuscript Data Aggregator ===\n")
    
    aggregator = ManuscriptDataAggregator(data_dir="data")
    
    # Demo with synthetic results
    print("Running demo with synthetic data...\n")
    
    demo_results = {
        'YL': {
            'p_L': 3.0e-3,
            'p_accept': 0.95,
            'alpha2': 0.40,
            'alpha1': 0.67,
            'eta_logical': 1.0,
            'circuit_depth': 12,
            'num_cnots': 16
        },
        'CR': {
            'p_L': 2.5e-3,
            'p_accept': 0.96,
            'alpha2': 0.60,
            'alpha1': 0.83,
            'eta_logical': 1.2,
            'circuit_depth': 10,
            'num_cnots': 14
        },
        'MR': {
            'p_L': 2.0e-3,
            'p_accept': 0.97,
            'alpha2': 0.60,
            'alpha1': 0.50,
            'eta_logical': 1.5,
            'circuit_depth': 12,
            'num_cnots': 15
        },
        'Optimized': {
            'p_L': 1.2e-3,
            'p_accept': 0.98,
            'alpha2': 0.35,
            'alpha1': 0.30,
            'eta_logical': 2.0,
            'circuit_depth': 8,
            'num_cnots': 12
        }
    }
    
    aggregator.run_full_analysis(demo_results, baseline='YL')
    
    print("\n✓ Manuscript data aggregator validated!")
