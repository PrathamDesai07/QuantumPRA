"""
Phase 1 Quick Reference Guide
==============================

Essential commands and usage patterns for the implemented simulation framework.
"""

# =============================================================================
# BASIC USAGE
# =============================================================================

# 1. Run demo (quick test of all protocols)
python run_simulations.py

# 2. Single protocol simulation
python run_simulations.py --protocol CR --distance 3 --samples 10000

# 3. Compare multiple protocols
python run_simulations.py --protocol CR MR --samples 50000

# 4. Custom noise parameters
python run_simulations.py --protocol MR --p1 1e-5 --p2 1e-3 --samples 10000

# 5. Parameter sweep
python run_simulations.py --protocol CR MR --sweep --samples 100000

# =============================================================================
# PYTHON API USAGE
# =============================================================================

from src.protocols.lao_criger import CRProtocol, MRProtocol
from src.protocols.li_protocol import LiProtocol
from src.noise_models.depolarizing import NoiseModel
from src.simulation.monte_carlo import MonteCarloSimulator

# Create protocol
protocol = CRProtocol(distance=3)  # or MRProtocol(3), LiProtocol(3, 7)

# Create noise model
noise = NoiseModel.ibm_like()  # or .ion_trap_like(), .biased(eta=100)

# Or custom noise:
noise = NoiseModel(p1=1e-4, p2=1e-3, p_init=1e-3, p_meas=1e-3)

# Run simulation
sim = MonteCarloSimulator(protocol, noise, verbose=True)
result = sim.run(n_samples=10000)

# Access results
print(f"Acceptance: {result.p_accept:.4f}")
print(f"Logical error rate: {result.p_logical:.4e}")
print(f"Logical bias: {result.logical_bias:.2f}")

# Save results
result.save("data/raw/my_result.json")

# Load results
from src.simulation.monte_carlo import SimulationResult
loaded = SimulationResult.load("data/raw/my_result.json")

# =============================================================================
# PARAMETER SWEEP
# =============================================================================

from src.simulation.monte_carlo import run_parameter_sweep

protocols = [CRProtocol(3), MRProtocol(3)]
p2_values = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]

results = run_parameter_sweep(
    protocols=protocols,
    p2_values=p2_values,
    p1_ratio=0.1,  # p1 = 0.1 * p2
    n_samples=100000,
    output_dir="data/raw"
)

# =============================================================================
# TESTING
# =============================================================================

# Run all unit tests
python -m pytest tests/test_phase1.py -v

# Run specific test class
python -m pytest tests/test_phase1.py::TestProtocols -v

# Run with coverage
python -m pytest tests/test_phase1.py --cov=src --cov-report=html

# =============================================================================
# AVAILABLE PROTOCOLS
# =============================================================================

# Li Protocol (2015) - Regular surface code
LiProtocol(distance_initial=3, distance_final=7)
# - Corner injection
# - Two-phase encoding
# - Leading-order: p_L ≈ (2/5) p2^2

# CR Protocol (2022) - Corner-rotated
CRProtocol(distance=3)
# - Corner injection on rotated surface code
# - Leading-order: p_L ≈ (3/5) p2^2

# MR Protocol (2022) - Middle-rotated  
MRProtocol(distance=3)
# - Middle injection on rotated surface code
# - Fewer sensitive qubits
# - Leading-order: p_L ≈ (3/5) p2^2 (better p1 coefficient)

# =============================================================================
# NOISE MODELS
# =============================================================================

# Preset configurations
NoiseModel.ibm_like()        # p1=1e-4, p2=1e-3, unbiased
NoiseModel.ion_trap_like()   # p1=1e-5, p2=1e-3, unbiased
NoiseModel.biased(eta=100)   # Dephasing-dominated, η=100

# Custom model
NoiseModel(
    p1=1e-4,          # Single-qubit gate error
    p2=1e-3,          # Two-qubit gate error
    p_init=1e-3,      # Initialization error
    p_meas=1e-3,      # Measurement error
    p_idle=1e-5,      # Idle error per timestep
    bias_eta=1.0      # Bias ratio (1.0 = unbiased)
)

# =============================================================================
# EXPECTED RESULTS (ROUGH ESTIMATES)
# =============================================================================

# For p2 = 1e-3, p1 = 1e-4 (IBM-like):
# - Acceptance rate: ~60%
# - Logical error rate: ~4e-4 (CR/MR), ~2e-4 (Li)
# - Runtime: ~10s for 10k samples

# For p2 = 1e-2, p1 = 1e-3:
# - Acceptance rate: ~40-50%
# - Logical error rate: ~4e-2
# - Runtime: ~10s for 10k samples

# =============================================================================
# FILE STRUCTURE
# =============================================================================

src/
├── protocols/
│   ├── base_protocol.py      # Abstract base class
│   ├── li_protocol.py        # Li (2015) YL protocol
│   └── lao_criger.py         # CR and MR protocols
├── noise_models/
│   └── depolarizing.py       # Noise channels
├── simulation/
│   ├── stabilizer_tableau.py # Stabilizer operations
│   └── monte_carlo.py        # MC simulator
tests/
└── test_phase1.py            # Unit tests
data/
├── raw/                      # Simulation outputs (JSON)
└── processed/                # Analysis results

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

# Import errors:
# - Make sure you're in the QuantumPRA directory
# - Check that __init__.py files exist in all src/ subdirectories

# Low acceptance rates (< 20%):
# - This is expected for high error rates (p2 > 5e-3)
# - Post-selection is working correctly

# All logical errors are X or all are Z:
# - Small sample size, increase n_samples
# - With 1000 samples, may see statistical fluctuations

# Slow performance:
# - Current implementation uses simplified circuit simulation
# - Full circuit-level simulation will be ~10-100x slower (Phase 2)
# - Use smaller n_samples for testing

# =============================================================================
# NEXT STEPS (PHASE 2)
# =============================================================================

# To implement full circuit-level simulation:
# 1. Complete circuit construction with detailed CNOT scheduling
# 2. Implement _execute_circuit_with_noise() in monte_carlo.py
# 3. Add logical operator measurement
# 4. Validate against analytical expressions

# To add analytic error analysis:
# 1. Implement Heisenberg-picture stabilizer tracking
# 2. Enumerate single-fault locations
# 3. Derive symbolic p_L expressions
# 4. Compare with numerical results
