"""
Figure Generation Tools for PRA Manuscript

This module generates publication-quality figures for the magic-state injection paper.
All figures follow Physical Review style guidelines:
- Vector graphics (PDF/EPS preferred)
- Minimum 10pt font size
- Color-blind friendly palettes
- Clear legends and labels

Main figures:
1. Protocol comparison: p_L vs noise parameters
2. Bias dependence: η_L vs η for different protocols
3. CNOT optimization: improvement factors
4. Convergence validation: numerical vs analytic
5. Distillation overhead: qubit-time volume comparison
6. Distance scaling: p_L and overhead vs d
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


# Physical Review A style configuration
def configure_matplotlib_for_pra():
    """Configure matplotlib for PRA-style figures."""
    # Use serif font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    # Line widths and marker sizes
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['axes.linewidth'] = 1.0
    
    # Figure size for 2-column format
    plt.rcParams['figure.figsize'] = (7.0, 5.0)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['savefig.bbox'] = 'tight'
    
    # Grid
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linewidth'] = 0.5


# Color-blind friendly palette (Okabe-Ito)
COLORS = {
    'YL': '#E69F00',       # Orange
    'CR': '#56B4E9',       # Sky blue
    'MR': '#009E73',       # Bluish green
    'Optimized': '#F0E442', # Yellow
    'Stacked': '#0072B2',  # Blue
    'Masked': '#D55E00',   # Vermillion
    'XZZX': '#CC79A7',     # Reddish purple
    'analytic': '#000000', # Black
    'numerical': '#999999' # Gray
}

MARKERS = {
    'YL': 'o',
    'CR': 's',
    'MR': '^',
    'Optimized': 'D',
    'Stacked': 'v',
    'Masked': '<',
    'XZZX': '>'
}


class FigureGenerator:
    """Generate all paper figures."""
    
    def __init__(self, output_dir: str = "data/figures"):
        """
        Initialize figure generator.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        configure_matplotlib_for_pra()
    
    def figure1_protocol_comparison(
        self,
        results_data: Dict[str, Dict],
        save: bool = True
    ) -> None:
        """
        Figure 1: Protocol comparison - p_L vs p_2 for different protocols.
        
        Shows leading-order scaling and numerical validation.
        
        Args:
            results_data: Dict with structure:
                {protocol_name: {'p2': [...], 'p_L': [...], 'analytic': [...]}}
            save: Whether to save figure
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        
        for protocol, data in results_data.items():
            p2_values = np.array(data['p2'])
            p_L_numerical = np.array(data['p_L'])
            p_L_analytic = np.array(data.get('analytic', p_L_numerical))
            
            # Plot numerical results
            ax.loglog(p2_values, p_L_numerical, 
                     marker=MARKERS.get(protocol, 'o'),
                     color=COLORS.get(protocol, 'black'),
                     label=protocol, linewidth=2, markersize=7)
            
            # Plot analytic line (dashed)
            if 'analytic' in data:
                ax.loglog(p2_values, p_L_analytic,
                         linestyle='--', color=COLORS.get(protocol, 'black'),
                         linewidth=1, alpha=0.7)
        
        # Reference line: p_L ~ p_2^2
        p2_ref = np.logspace(-5, -2, 50)
        ax.loglog(p2_ref, 0.5 * p2_ref**2, 'k:', linewidth=1.5, 
                 label=r'$p_L \sim p_2^2$', alpha=0.5)
        
        ax.set_xlabel(r'Two-qubit gate error rate $p_2$', fontsize=11)
        ax.set_ylabel(r'Logical error rate $p_L$', fontsize=11)
        ax.set_title('Magic State Injection: Protocol Comparison', fontsize=12)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, which='both', alpha=0.3)
        
        if save:
            save_path = self.output_dir / "fig1_protocol_comparison.pdf"
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure 1 saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def figure2_bias_dependence(
        self,
        results_data: Dict[str, Dict],
        save: bool = True
    ) -> None:
        """
        Figure 2: Logical bias η_L vs physical bias η.
        
        Shows how protocols amplify or suppress bias.
        
        Args:
            results_data: Dict with structure:
                {protocol_name: {'eta_phys': [...], 'eta_logical': [...]}}
            save: Whether to save figure
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        
        for protocol, data in results_data.items():
            eta_phys = np.array(data['eta_phys'])
            eta_logical = np.array(data['eta_logical'])
            
            ax.loglog(eta_phys, eta_logical,
                     marker=MARKERS.get(protocol, 'o'),
                     color=COLORS.get(protocol, 'black'),
                     label=protocol, linewidth=2, markersize=7)
        
        # Reference: η_L = η (no bias change)
        eta_ref = np.logspace(0, 3, 50)
        ax.loglog(eta_ref, eta_ref, 'k--', linewidth=1.5, 
                 label=r'$\eta_L = \eta$ (no change)', alpha=0.5)
        
        ax.set_xlabel(r'Physical noise bias $\eta = p_Z / p_X$', fontsize=11)
        ax.set_ylabel(r'Logical noise bias $\eta_L$', fontsize=11)
        ax.set_title('Logical Bias vs Physical Bias', fontsize=12)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, which='both', alpha=0.3)
        
        if save:
            save_path = self.output_dir / "fig2_bias_dependence.pdf"
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure 2 saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def figure3_cnot_optimization(
        self,
        results_data: Dict[str, Dict],
        save: bool = True
    ) -> None:
        """
        Figure 3: CNOT ordering optimization results.
        
        Bar chart showing improvement factors for different stabilizers.
        
        Args:
            results_data: Dict with structure:
                {stabilizer_id: {'baseline': float, 'optimized': float, 'improvement': float}}
            save: Whether to save figure
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        stabilizer_ids = list(results_data.keys())
        improvements = [results_data[sid]['improvement'] for sid in stabilizer_ids]
        
        x_pos = np.arange(len(stabilizer_ids))
        bars = ax.bar(x_pos, improvements, color='#0072B2', alpha=0.8, edgecolor='black')
        
        # Color bars by improvement level
        for i, bar in enumerate(bars):
            if improvements[i] > 2.0:
                bar.set_color('#009E73')  # Green for >2x
            elif improvements[i] > 1.5:
                bar.set_color('#E69F00')  # Orange for 1.5-2x
        
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stabilizer_ids, rotation=45, ha='right')
        ax.set_xlabel('Stabilizer', fontsize=11)
        ax.set_ylabel('Improvement Factor', fontsize=11)
        ax.set_title('CNOT Ordering Optimization (Biased Noise)', fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#009E73', label='>2× improvement'),
            Patch(facecolor='#E69F00', label='1.5-2× improvement'),
            Patch(facecolor='#0072B2', label='<1.5× improvement')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        if save:
            save_path = self.output_dir / "fig3_cnot_optimization.pdf"
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure 3 saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def figure4_convergence_validation(
        self,
        convergence_data: Dict[str, Dict],
        save: bool = True
    ) -> None:
        """
        Figure 4: Convergence validation - p_L/p_2^2 vs p_2.
        
        Shows numerical results converging to analytic leading-order coefficient.
        
        Args:
            convergence_data: Dict with structure:
                {protocol: {'p2': [...], 'ratio': [...], 'alpha2_theory': float}}
            save: Whether to save figure
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        
        for protocol, data in convergence_data.items():
            p2_values = np.array(data['p2'])
            ratio = np.array(data['ratio'])
            alpha2_theory = data['alpha2_theory']
            
            ax.semilogx(p2_values, ratio,
                       marker=MARKERS.get(protocol, 'o'),
                       color=COLORS.get(protocol, 'black'),
                       label=protocol, linewidth=2, markersize=7)
            
            # Horizontal line at theory value
            ax.axhline(y=alpha2_theory, color=COLORS.get(protocol, 'black'),
                      linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel(r'Two-qubit gate error rate $p_2$', fontsize=11)
        ax.set_ylabel(r'$p_L / p_2^2$ (normalized)', fontsize=11)
        ax.set_title(r'Leading-Order Convergence: $p_L / p_2^2 \to \alpha_2$', fontsize=12)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        if save:
            save_path = self.output_dir / "fig4_convergence_validation.pdf"
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure 4 saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def figure5_distillation_overhead(
        self,
        overhead_data: Dict[str, Dict],
        save: bool = True
    ) -> None:
        """
        Figure 5: Distillation overhead comparison.
        
        Bar chart comparing qubit-time volume for different protocols.
        
        Args:
            overhead_data: Dict with structure:
                {protocol: {'volume': float, 'rounds': int, 'raw_states': float}}
            save: Whether to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        protocols = list(overhead_data.keys())
        volumes = [overhead_data[p]['volume'] for p in protocols]
        raw_states = [overhead_data[p]['raw_states'] for p in protocols]
        
        # Normalize to baseline
        baseline_volume = volumes[0]
        relative_volumes = [v / baseline_volume for v in volumes]
        
        x_pos = np.arange(len(protocols))
        
        # Panel 1: Relative qubit-time volume
        bars1 = ax1.bar(x_pos, relative_volumes, 
                       color=[COLORS.get(p, '#999999') for p in protocols],
                       alpha=0.8, edgecolor='black')
        ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(protocols, rotation=45, ha='right')
        ax1.set_ylabel('Relative Overhead', fontsize=11)
        ax1.set_title('Qubit-Time Volume (normalized to YL)', fontsize=11)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Panel 2: Raw magic states consumed
        bars2 = ax2.bar(x_pos, raw_states,
                       color=[COLORS.get(p, '#999999') for p in protocols],
                       alpha=0.8, edgecolor='black')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(protocols, rotation=45, ha='right')
        ax2.set_ylabel('Raw States per T gate', fontsize=11)
        ax2.set_yscale('log')
        ax2.set_title('Raw Magic State Consumption', fontsize=11)
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "fig5_distillation_overhead.pdf"
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure 5 saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def figure6_distance_scaling(
        self,
        scaling_data: Dict[str, Dict],
        save: bool = True
    ) -> None:
        """
        Figure 6: p_L scaling with code distance.
        
        Shows how error rates change with distance for each protocol.
        
        Args:
            scaling_data: Dict with structure:
                {protocol: {'distances': [...], 'p_L': [...], 'overhead': [...]}}
            save: Whether to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        for protocol, data in scaling_data.items():
            distances = np.array(data['distances'])
            p_L_values = np.array(data['p_L'])
            overheads = np.array(data['overhead'])
            
            ax1.semilogy(distances, p_L_values,
                        marker=MARKERS.get(protocol, 'o'),
                        color=COLORS.get(protocol, 'black'),
                        label=protocol, linewidth=2, markersize=7)
            
            ax2.semilogy(distances, overheads,
                        marker=MARKERS.get(protocol, 'o'),
                        color=COLORS.get(protocol, 'black'),
                        label=protocol, linewidth=2, markersize=7)
        
        ax1.set_xlabel('Code Distance $d$', fontsize=11)
        ax1.set_ylabel(r'Injection Error Rate $p_L$', fontsize=11)
        ax1.set_title('Error Rate vs Distance', fontsize=11)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Code Distance $d$', fontsize=11)
        ax2.set_ylabel('Qubit-Time Volume', fontsize=11)
        ax2.set_title('Overhead vs Distance', fontsize=11)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "fig6_distance_scaling.pdf"
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure 6 saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_all_figures(
        self,
        data_file: str,
        save: bool = True
    ) -> None:
        """
        Generate all paper figures from aggregated data file.
        
        Args:
            data_file: Path to JSON file with all results
            save: Whether to save figures
        """
        with open(data_file, 'r') as f:
            all_data = json.load(f)
        
        print("Generating all figures...")
        
        if 'protocol_comparison' in all_data:
            self.figure1_protocol_comparison(all_data['protocol_comparison'], save)
        
        if 'bias_dependence' in all_data:
            self.figure2_bias_dependence(all_data['bias_dependence'], save)
        
        if 'cnot_optimization' in all_data:
            self.figure3_cnot_optimization(all_data['cnot_optimization'], save)
        
        if 'convergence' in all_data:
            self.figure4_convergence_validation(all_data['convergence'], save)
        
        if 'distillation' in all_data:
            self.figure5_distillation_overhead(all_data['distillation'], save)
        
        if 'distance_scaling' in all_data:
            self.figure6_distance_scaling(all_data['distance_scaling'], save)
        
        print(f"\nAll figures generated in {self.output_dir}/")


# Example usage
if __name__ == "__main__":
    print("=== Figure Generator for PRA Manuscript ===\n")
    
    generator = FigureGenerator(output_dir="data/figures")
    
    # Demo with synthetic data
    print("Generating demo figures with synthetic data...\n")
    
    # Figure 1 demo
    p2_demo = np.logspace(-5, -2, 20)
    demo_data_fig1 = {
        'YL': {
            'p2': p2_demo.tolist(),
            'p_L': (0.4 * p2_demo**2).tolist(),
            'analytic': (0.4 * p2_demo**2).tolist()
        },
        'CR': {
            'p2': p2_demo.tolist(),
            'p_L': (0.6 * p2_demo**2).tolist(),
            'analytic': (0.6 * p2_demo**2).tolist()
        },
        'MR': {
            'p2': p2_demo.tolist(),
            'p_L': (0.5 * p2_demo**2).tolist(),
            'analytic': (0.5 * p2_demo**2).tolist()
        }
    }
    generator.figure1_protocol_comparison(demo_data_fig1, save=True)
    
    print("✓ Figure generation module validated!")
    print(f"✓ Demo figures saved in {generator.output_dir}/")
