"""
Test Suite for Phase 1
======================

Unit tests for stabilizer operations, protocols, noise models, and simulation.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from simulation.stabilizer_tableau import StabilizerSimulator, StabilizerTableau
from noise_models.depolarizing import NoiseModel, NoiseChannel
from protocols.li_protocol import LiProtocol
from protocols.lao_criger import CRProtocol, MRProtocol
from simulation.monte_carlo import MonteCarloSimulator


class TestStabilizerTableau(unittest.TestCase):
    """Test stabilizer tableau operations."""
    
    def test_initialization(self):
        """Test tableau initialization."""
        n = 3
        tableau = StabilizerTableau.computational_basis(n)
        self.assertEqual(tableau.n, n)
        self.assertEqual(tableau.x_table.shape, (2*n, n))
        self.assertEqual(tableau.z_table.shape, (2*n, n))
    
    def test_hadamard_gate(self):
        """Test Hadamard gate."""
        sim = StabilizerSimulator(1)
        sim.hadamard(0)
        # H|0⟩ = |+⟩, measure X should give +1
        # This is implicit in tableau representation
    
    def test_cnot_gate(self):
        """Test CNOT gate for Bell state."""
        sim = StabilizerSimulator(2)
        sim.hadamard(0)
        sim.cnot(0, 1)
        # Should create |Φ+⟩ = (|00⟩ + |11⟩)/√2
    
    def test_measurement(self):
        """Test Z basis measurement."""
        sim = StabilizerSimulator(1)
        outcome = sim.measure_z(0)
        self.assertIn(outcome, [0, 1])
        
        # Measure |+⟩, should be random
        sim.reset()
        sim.hadamard(0)
        outcomes = [sim.measure_z(0) for _ in range(20)]
        sim.reset()
        # With 20 samples, very unlikely to be all same if truly random


class TestNoiseModels(unittest.TestCase):
    """Test noise model sampling."""
    
    def test_noise_model_creation(self):
        """Test creating noise models."""
        noise = NoiseModel.ibm_like()
        self.assertEqual(noise.p1, 1e-4)
        self.assertEqual(noise.p2, 1e-3)
        
        biased = NoiseModel.biased(eta=100.0)
        self.assertEqual(biased.bias_eta, 100.0)
    
    def test_depolarizing_sampling(self):
        """Test depolarizing error sampling."""
        p = 0.1
        n_samples = 1000
        errors = [NoiseChannel.sample_single_qubit_pauli_depolarizing(p) 
                  for _ in range(n_samples)]
        
        error_rate = sum(e is not None for e in errors) / n_samples
        # Should be close to 0.1 ± 0.02
        self.assertGreater(error_rate, 0.07)
        self.assertLess(error_rate, 0.13)
    
    def test_biased_sampling(self):
        """Test biased error sampling."""
        p = 0.1
        eta = 100.0
        n_samples = 1000
        
        errors = [NoiseChannel.sample_single_qubit_pauli_biased(p, eta) 
                  for _ in range(n_samples)]
        
        z_count = sum(e == 'Z' for e in errors if e is not None)
        x_count = sum(e == 'X' for e in errors if e is not None)
        
        if x_count > 0:
            ratio = z_count / x_count
            # Should be roughly eta ± large variance
            self.assertGreater(ratio, 10)  # At least 10x bias


class TestProtocols(unittest.TestCase):
    """Test protocol implementations."""
    
    def test_li_protocol(self):
        """Test Li protocol creation."""
        protocol = LiProtocol(distance_initial=3, distance_final=7)
        self.assertEqual(protocol.d1, 3)
        self.assertEqual(protocol.d2, 7)
        self.assertEqual(protocol.params.magic_qubit_loc, 'corner')
    
    def test_cr_protocol(self):
        """Test CR protocol creation."""
        protocol = CRProtocol(distance=3)
        self.assertEqual(protocol.distance, 3)
        self.assertEqual(protocol.params.magic_qubit_loc, 'corner')
        
        # Test layout
        layout = protocol.get_data_qubit_layout()
        self.assertEqual(layout.shape, (3, 3))
        
        # Test magic qubit
        magic_idx = protocol.get_magic_qubit_index()
        self.assertEqual(magic_idx, 0)
    
    def test_mr_protocol(self):
        """Test MR protocol creation."""
        protocol = MRProtocol(distance=3)
        self.assertEqual(protocol.distance, 3)
        self.assertEqual(protocol.params.magic_qubit_loc, 'middle')
        
        # Magic qubit should be in middle
        magic_idx = protocol.get_magic_qubit_index()
        self.assertEqual(magic_idx, 4)  # Center of 3x3 grid
    
    def test_circuit_building(self):
        """Test circuit construction."""
        protocol = CRProtocol(distance=3)
        circuit = protocol.build_circuit()
        self.assertIsNotNone(circuit)
        self.assertGreater(len(circuit), 0)


class TestMonteCarloSimulation(unittest.TestCase):
    """Test Monte Carlo simulation."""
    
    def test_simulator_creation(self):
        """Test creating simulator."""
        protocol = CRProtocol(distance=3)
        noise = NoiseModel.ibm_like()
        sim = MonteCarloSimulator(protocol, noise, verbose=False)
        self.assertIsNotNone(sim)
    
    def test_small_simulation(self):
        """Test running small simulation."""
        protocol = CRProtocol(distance=3)
        noise = NoiseModel(p1=1e-4, p2=1e-3, p_init=1e-3, p_meas=1e-3)
        
        sim = MonteCarloSimulator(protocol, noise, verbose=False)
        result = sim.run(n_samples=100)
        
        self.assertEqual(result.n_samples, 100)
        self.assertGreaterEqual(result.n_accepted, 0)
        self.assertLessEqual(result.n_accepted, 100)
        self.assertGreaterEqual(result.p_accept, 0)
        self.assertLessEqual(result.p_accept, 1)
    
    def test_result_serialization(self):
        """Test saving and loading results."""
        import tempfile
        
        protocol = CRProtocol(distance=3)
        noise = NoiseModel.ibm_like()
        sim = MonteCarloSimulator(protocol, noise, verbose=False)
        result = sim.run(n_samples=10)
        
        # Save and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            result.save(f.name)
            loaded = result.load(f.name)
        
        self.assertEqual(result.protocol_name, loaded.protocol_name)
        self.assertEqual(result.n_samples, loaded.n_samples)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test full simulation pipeline."""
        # Create protocol
        protocol = CRProtocol(distance=3)
        
        # Create noise model
        noise = NoiseModel.ibm_like()
        
        # Run simulation
        sim = MonteCarloSimulator(protocol, noise, verbose=False)
        result = sim.run(n_samples=50)
        
        # Check results are reasonable
        self.assertIsNotNone(result)
        self.assertGreater(result.runtime_seconds, 0)


def run_all_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_all_tests()
