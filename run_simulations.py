"""
Main Driver Script for Magic-State Injection Simulations
=========================================================

Entry point for running Phase 1 simulations.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from protocols.li_protocol import LiProtocol
from protocols.lao_criger import CRProtocol, MRProtocol
from noise_models.depolarizing import NoiseModel
from simulation.monte_carlo import MonteCarloSimulator, run_parameter_sweep


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run magic-state injection simulations'
    )
    
    parser.add_argument(
        '--protocol',
        type=str,
        nargs='+',
        default=['CR'],
        choices=['Li', 'CR', 'MR'],
        help='Protocols to simulate (default: CR)'
    )
    
    parser.add_argument(
        '--distance',
        type=int,
        default=3,
        help='Code distance (default: 3)'
    )
    
    parser.add_argument(
        '--p1',
        type=float,
        default=1e-4,
        help='Single-qubit gate error rate (default: 1e-4)'
    )
    
    parser.add_argument(
        '--p2',
        type=float,
        default=1e-3,
        help='Two-qubit gate error rate (default: 1e-3)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=10000,
        help='Number of Monte Carlo samples (default: 10000)'
    )
    
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Run parameter sweep over p2 values'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw',
        help='Output directory for results (default: data/raw)'
    )
    
    args = parser.parse_args()
    
    # Create protocols
    protocols = []
    for prot_name in args.protocol:
        if prot_name == 'Li':
            protocols.append(LiProtocol(distance_initial=3, distance_final=args.distance))
        elif prot_name == 'CR':
            protocols.append(CRProtocol(distance=args.distance))
        elif prot_name == 'MR':
            protocols.append(MRProtocol(distance=args.distance))
    
    if args.sweep:
        # Parameter sweep
        p2_values = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
        print(f"\nRunning parameter sweep over {len(p2_values)} p2 values...")
        print(f"Protocols: {[p.params.protocol_name for p in protocols]}")
        print(f"Samples per point: {args.samples}")
        
        results = run_parameter_sweep(
            protocols=protocols,
            p2_values=p2_values,
            p1_ratio=args.p1 / args.p2,
            n_samples=args.samples,
            output_dir=args.output
        )
        
        print(f"\nSweep complete! Results saved to {args.output}/")
        
    else:
        # Single simulation
        noise = NoiseModel(
            p1=args.p1,
            p2=args.p2,
            p_init=args.p2,
            p_meas=args.p2,
            p_idle=0.0
        )
        
        for protocol in protocols:
            print(f"\n{'='*70}")
            print(f"Simulating {protocol.params.protocol_name}")
            print(f"{'='*70}")
            
            sim = MonteCarloSimulator(protocol, noise, verbose=True)
            result = sim.run(n_samples=args.samples)
            
            # Save result
            filename = f"{protocol.params.protocol_name}_p2{args.p2:.1e}.json"
            filepath = Path(args.output) / filename
            result.save(str(filepath))
            print(f"\nResult saved to {filepath}")


def demo():
    """Run a quick demo simulation."""
    print("="*70)
    print("MAGIC-STATE INJECTION SIMULATION - DEMO MODE")
    print("="*70)
    print("\nPhase 1 Implementation: Core Features")
    print("- Stabilizer tableau operations")
    print("- Circuit-level noise models")
    print("- CR, MR, and Li protocols")
    print("- Monte Carlo simulation framework")
    print()
    
    # Quick comparison of CR and MR
    protocols = [
        CRProtocol(distance=3),
        MRProtocol(distance=3)
    ]
    
    noise = NoiseModel.ibm_like()
    
    for protocol in protocols:
        print(f"\nRunning {protocol.params.protocol_name}...")
        sim = MonteCarloSimulator(protocol, noise, verbose=True)
        result = sim.run(n_samples=1000)
        print()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # No arguments: run demo
        demo()
    else:
        main()
