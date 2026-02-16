# Phase 2 Quick Reference

**Quick guide to analytic analysis modules**

## Module Overview

```
src/analysis/
├── circuit_builder.py          # Build detailed quantum circuits
├── error_enumeration.py        # Enumerate single-fault locations
├── symbolic_expressions.py     # Derive closed-form p_L expressions
├── logical_bias.py             # Analyze logical noise bias
└── compare_analytic_numerical.py  # Validate analytic vs numerical
```

## Quick Start Examples

### 1. Symbolic p_L Expressions

```python
from analysis.symbolic_expressions import SymbolicAnalyzer

# Create analyzer
analyzer = SymbolicAnalyzer()

# Derive expressions for each protocol
expr_YL = analyzer.derive_YL_expression()
expr_CR = analyzer.derive_CR_expression()
expr_MR = analyzer.derive_MR_expression()

# Evaluate at specific parameters
p_L = expr_CR.evaluate(p_1=1e-4, p_2=1e-3, p_I=1e-3, p_M=1e-3)
print(f"p_L = {p_L:.4e}")

# Get coefficients
print(f"Leading coefficient α₂ = {expr_CR.coefficients['alpha_2']}")

# Compare protocols
comparison = analyzer.compare_protocols(['YL', 'CR', 'MR'], p_1=1e-4, p_2=1e-3)
for prot, p_L in comparison.items():
    print(f"{prot}: {p_L:.4e}")
```

### 2. Fault Enumeration

```python
from analysis.error_enumeration import FaultEnumerator, FaultLocation
from simulation.circuit_builder import build_distance3_cr_circuit

# Build circuit
circuit = build_distance3_cr_circuit()

# Create enumerator
enumerator = FaultEnumerator(circuit)

# Enumerate all faults
faults = enumerator.enumerate_all_faults()
print(f"Total faults: {len(faults)}")

# Count by location
for loc in FaultLocation:
    count = len([f for f in faults if f.location == loc])
    print(f"{loc.name}: {count}")

# Classify faults (requires Heisenberg tracking)
# detected, undetected = enumerator.classify_faults(faults, stabilizers)
# p_L = enumerator.compute_leading_order_pL(undetected, noise_params)
```

### 3. Logical Bias Analysis

```python
from analysis.logical_bias import LogicalBiasAnalyzer

# Create analyzer
analyzer = LogicalBiasAnalyzer()

# Analyze bias propagation
result = analyzer.analyze_bias_propagation(
    protocol_name='MR',
    p2=1e-3,
    eta=100,  # Physical bias η = p_Z/p_X
    distance=3
)

print(f"Physical bias: {result.physical_bias:.1f}")
print(f"Logical bias: {result.logical_bias:.1f}")
print(f"p_X̄: {result.p_logical_x:.4e}")
print(f"p_Z̄: {result.p_logical_z:.4e}")

# Compute distillation benefit
benefit = analyzer.compute_distillation_benefit(
    protocol_name='MR',
    p2_range=np.logspace(-4, -2, 10),
    eta=100,
    distance=3
)

print(f"Overhead reduction: {benefit['overhead_reduction']:.1%}")
```

### 4. Analytic vs Numerical Comparison

```python
from analysis.compare_analytic_numerical import AnalyticNumericalComparator
from simulation.monte_carlo import MonteCarloSimulator, SimulationResult

# Create comparator
comparator = AnalyticNumericalComparator()

# Compare single point (with pre-existing simulation result)
result = comparator.compare_single_point(
    protocol_name='CR',
    p1=1e-4,
    p2=1e-3,
    p_init=1e-3,
    p_meas=1e-3,
    sim_result=sim_result  # From MonteCarloSimulator.run()
)

print(f"Analytic: {result.p_L_analytic:.4e}")
print(f"Numerical: {result.p_L_numerical:.4e}")
print(f"Relative error: {result.relative_error:.1%}")
print(f"Agreement: {'✓' if result.agrees_within_tolerance() else '✗'}")

# Compare parameter sweep
p2_values = np.logspace(-4, -2, 10)
results = comparator.compare_sweep(
    protocol_name='CR',
    p2_values=p2_values,
    p1_ratio=0.1,
    simulation_dir="data/raw"
)

# Generate plots
comparator.plot_comparison(
    protocol_names=['CR', 'MR'],
    p2_range=p2_values,
    save_path='figures/validation.png'
)

# Generate validation report
comparator.generate_validation_report('data/processed/validation_report.txt')
```

### 5. Circuit Builder

```python
from simulation.circuit_builder import CircuitBuilder, build_distance3_cr_circuit, Gate

# Use pre-built circuit
circuit = build_distance3_cr_circuit()
print(f"Timesteps: {len(circuit.timesteps)}")
print(f"Total gates: {sum(len(ts.gates) for ts in circuit.timesteps)}")

# Build custom circuit
builder = CircuitBuilder(n_qubits=13)

# Add initialization
for i in range(13):
    builder.add_gate(Gate('INIT', ['Z'], [i]))

# Add syndrome measurement round
builder.add_cnot(control=0, target=1)
builder.add_cnot(control=2, target=1)
# ... more CNOTs ...

# Measure syndrome qubits
for i in [1, 3, 5, 7]:
    builder.add_gate(Gate('MEASURE', ['Z'], [i]))

# Get circuit
custom_circuit = builder.get_circuit()
```

## Key Formulas

### Leading-Order p_L Expressions

**YL Protocol (Li 2015)**:
```
p_L ≈ (2/5) p₂² + (2/3) p₁ + 2 p_I
```

**CR Protocol (Lao-Criger 2022, corner)**:
```
p_L ≈ (3/5) p₂² + (5/6) p₁ + 2 p_I
```

**MR Protocol (Lao-Criger 2022, middle)**:
```
p_L ≈ (3/5) p₂² + (1/2) p₁ + p_I
```

### Logical Bias

```
η_L = p_Z̄ / p_X̄
```

For biased noise (η >> 1), protocols can be optimized to suppress the dominant error channel.

### Relative Error

```
relative_error = |p_L_analytic - p_L_numerical| / p_L_numerical
```

Typical agreement within 20% for leading-order expressions at low noise.

## Testing

```bash
# Run Phase 1 tests
pytest tests/test_phase1.py -v

# Run comparison module test
python src/analysis/compare_analytic_numerical.py

# Run all tests
pytest tests/ -v
```

## Validation Workflow

1. **Run Monte Carlo simulation**:
   ```python
   from simulation.monte_carlo import MonteCarloSimulator
   from protocols.lao_criger import CRProtocol
   from noise_models.depolarizing import NoiseModel
   
   protocol = CRProtocol()
   noise = NoiseModel.ibm_like()
   simulator = MonteCarloSimulator(protocol, noise)
   result = simulator.run(distance=3, n_samples=100000)
   result.save('data/raw/CR_d3_p2_1e3.json')
   ```

2. **Compare with analytic**:
   ```python
   from analysis.compare_analytic_numerical import AnalyticNumericalComparator
   
   comparator = AnalyticNumericalComparator()
   comparison = comparator.compare_single_point(
       protocol_name='CR',
       p1=1e-4, p2=1e-3, p_init=1e-3, p_meas=1e-3,
       sim_result=result
   )
   ```

3. **Generate validation report**:
   ```python
   comparator.generate_validation_report('validation_report.txt')
   ```

## Common Parameters

```python
# IBM-like regime
p1 = 1e-4
p2 = 1e-3
p_init = 1e-3
p_meas = 1e-3
bias_eta = 1.0  # unbiased

# Ion trap-like regime  
p1 = 1e-5
p2 = 1e-3
p_init = 1e-4
p_meas = 1e-4
bias_eta = 1.0

# Aggressively biased regime
p1 = 1e-5
p2 = 1e-3
p_init = 1e-4
p_meas = 1e-3
bias_eta = 100.0  # dephasing-dominated
```

## File I/O

```python
# Save simulation results
result.save('data/raw/protocol_d3.json')

# Load simulation results
from simulation.monte_carlo import SimulationResult
result = SimulationResult.load('data/raw/protocol_d3.json')

# Save comparison data
import json
with open('data/processed/comparison.json', 'w') as f:
    json.dump(comparison_data, f)
```

## Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Plot p_L vs p2
p2_range = np.logspace(-4, -2, 20)
p_L_values = [expr.evaluate(p_1=0.1*p2, p_2=p2, p_I=p2, p_M=p2) 
              for p2 in p2_range]

plt.figure(figsize=(8, 6))
plt.loglog(p2_range, p_L_values, 'o-', label='Analytic')
plt.xlabel('Two-qubit error rate $p_2$')
plt.ylabel('Logical error rate $p_L$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('figures/p_L_vs_p2.png', dpi=150)
```

## Troubleshooting

**Q: SymPy expression won't evaluate**  
A: Check that all required parameters (p_1, p_2, p_I, p_M) are provided. Missing parameters will cause evaluation to fail.

**Q: Fault enumeration returns empty list**  
A: Ensure circuit has been built properly with `CircuitBuilder` and has gates/timesteps.

**Q: Comparison shows large relative error**  
A: Expected for leading-order expressions. Higher-order terms become important at larger p2. For p2 > 1e-2, consider second-order expressions.

**Q: Monte Carlo gives different results each time**  
A: This is normal statistical variation. Use more samples (n_samples > 1e5) for better precision.

---

For detailed documentation, see:
- [PHASE2_COMPLETE.md](PHASE2_COMPLETE.md) - Full Phase 2 summary
- [PHASE1_COMPLETE.md](PHASE1_COMPLETE.md) - Phase 1 summary
- [README.md](../README.md) - Main project README
