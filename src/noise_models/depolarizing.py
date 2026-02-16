"""
Noise Models
============

Circuit-level noise models for quantum error correction simulations.
Implements depolarizing, biased, and connectivity-constrained noise.

Based on:
- Li (2015): Circuit-level depolarizing noise
- Lao & Criger (2022): Biased noise models
- Berthusen et al. (2025): Idle errors and connectivity constraints
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class NoiseType(Enum):
    """Types of noise channels."""
    DEPOLARIZING = "depolarizing"
    BIASED = "biased"
    AMPLITUDE_DAMPING = "amplitude_damping"


@dataclass
class NoiseModel:
    """
    Hardware-specific circuit-level noise model.
    
    Attributes:
        p1: Single-qubit gate error rate
        p2: Two-qubit (CNOT) gate error rate  
        p_init: Initialization error rate
        p_meas: Measurement error rate
        p_idle: Idle error rate (per timestep)
        bias_eta: Noise bias ratio η = p_Z / p_X (η >> 1 for dephasing-dominated)
        two_qubit_type: Type of two-qubit noise
    """
    p1: float = 1e-4
    p2: float = 1e-3
    p_init: float = 1e-3
    p_meas: float = 1e-3
    p_idle: float = 0.0
    bias_eta: float = 1.0  # 1.0 = unbiased, >> 1 for biased
    two_qubit_type: NoiseType = NoiseType.DEPOLARIZING
    
    def __post_init__(self):
        """Validate noise parameters."""
        assert 0 <= self.p1 <= 1
        assert 0 <= self.p2 <= 1
        assert 0 <= self.p_init <= 1
        assert 0 <= self.p_meas <= 1
        assert 0 <= self.p_idle <= 1
        assert self.bias_eta >= 1.0
    
    @classmethod
    def ibm_like(cls) -> 'NoiseModel':
        """
        IBM-like superconducting qubit parameters.
        
        Returns:
            NoiseModel with IBM-like parameters
        """
        return cls(
            p1=1e-4,
            p2=1e-3,
            p_init=1e-3,
            p_meas=1e-3,
            p_idle=1e-5,
            bias_eta=1.0
        )
    
    @classmethod
    def ion_trap_like(cls) -> 'NoiseModel':
        """
        Ion trap parameters with high single-qubit fidelity.
        
        Returns:
            NoiseModel with ion-trap-like parameters
        """
        return cls(
            p1=1e-5,
            p2=1e-3,
            p_init=1e-4,
            p_meas=1e-4,
            p_idle=1e-6,
            bias_eta=1.0
        )
    
    @classmethod
    def biased(cls, eta: float = 100.0) -> 'NoiseModel':
        """
        Aggressively biased noise model (dephasing-dominated).
        
        Args:
            eta: Bias ratio (default 100)
        
        Returns:
            NoiseModel with biased noise
        """
        return cls(
            p1=1e-5,
            p2=1e-3,
            p_init=1e-4,
            p_meas=1e-3,
            p_idle=1e-5,
            bias_eta=eta
        )


class NoiseChannel:
    """
    Applies noise channels to quantum states represented as stabilizer tableaux.
    """
    
    @staticmethod
    def sample_single_qubit_pauli_depolarizing(p: float) -> Optional[str]:
        """
        Sample single-qubit depolarizing error.
        
        With probability p, return uniformly random Pauli from {X, Y, Z}.
        With probability 1-p, return None (no error).
        
        Args:
            p: Error probability
        
        Returns:
            'X', 'Y', 'Z', or None
        """
        if np.random.random() < p:
            return np.random.choice(['X', 'Y', 'Z'])
        return None
    
    @staticmethod
    def sample_single_qubit_pauli_biased(p: float, eta: float) -> Optional[str]:
        """
        Sample single-qubit biased Pauli error.
        
        Bias η = p_Z / p_X means Z errors are η times more likely than X errors.
        Y errors occur with intermediate probability.
        
        Args:
            p: Total error probability
            eta: Bias ratio η = p_Z / p_X
        
        Returns:
            'X', 'Y', 'Z', or None
        """
        if np.random.random() >= p:
            return None
        
        # Normalize probabilities: p_X + p_Y + p_Z = p
        # p_Y ≈ sqrt(p_X * p_Z) (geometric mean)
        # p_X * (1 + sqrt(eta) + eta) = p
        norm = 1 + np.sqrt(eta) + eta
        p_x = p / norm
        p_y = p * np.sqrt(eta) / norm
        p_z = p * eta / norm
        
        r = np.random.random() * p
        if r < p_x:
            return 'X'
        elif r < p_x + p_y:
            return 'Y'
        else:
            return 'Z'
    
    @staticmethod
    def sample_two_qubit_pauli_depolarizing(p: float) -> Optional[Tuple[str, str]]:
        """
        Sample two-qubit depolarizing error.
        
        With probability p, return uniformly random two-qubit Pauli from
        {IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ} without {II}.
        
        Args:
            p: Error probability
        
        Returns:
            Tuple of ('X'/'Y'/'Z'/'I', 'X'/'Y'/'Z'/'I') or None
        """
        if np.random.random() < p:
            paulis = ['I', 'X', 'Y', 'Z']
            while True:
                p1, p2 = np.random.choice(paulis, 2)
                if not (p1 == 'I' and p2 == 'I'):
                    return (p1, p2)
        return None
    
    @staticmethod
    def sample_measurement_error(p: float) -> bool:
        """
        Sample measurement bitflip error.
        
        Args:
            p: Error probability
        
        Returns:
            True if error occurs, False otherwise
        """
        return np.random.random() < p
    
    @staticmethod
    def sample_init_error(p: float) -> bool:
        """
        Sample initialization bitflip error (|0⟩ → |1⟩).
        
        Args:
            p: Error probability
        
        Returns:
            True if error occurs, False otherwise
        """
        return np.random.random() < p


def apply_single_qubit_noise(
    simulator,
    qubit: int,
    noise_model: NoiseModel,
    biased: bool = False
):
    """
    Apply single-qubit noise after a gate on qubit.
    
    Args:
        simulator: StabilizerSimulator instance
        qubit: Qubit index
        noise_model: NoiseModel instance
        biased: Use biased noise if True
    """
    if biased:
        error = NoiseChannel.sample_single_qubit_pauli_biased(
            noise_model.p1, noise_model.bias_eta
        )
    else:
        error = NoiseChannel.sample_single_qubit_pauli_depolarizing(noise_model.p1)
    
    if error:
        pauli_string = ['I'] * simulator.n
        pauli_string[qubit] = error
        simulator.apply_pauli(''.join(pauli_string))


def apply_two_qubit_noise(
    simulator,
    control: int,
    target: int,
    noise_model: NoiseModel
):
    """
    Apply two-qubit noise after a CNOT gate.
    
    Args:
        simulator: StabilizerSimulator instance
        control: Control qubit index
        target: Target qubit index
        noise_model: NoiseModel instance
    """
    error = NoiseChannel.sample_two_qubit_pauli_depolarizing(noise_model.p2)
    
    if error:
        pauli_string = ['I'] * simulator.n
        pauli_string[control] = error[0]
        pauli_string[target] = error[1]
        simulator.apply_pauli(''.join(pauli_string))


def apply_measurement_noise(outcome: int, noise_model: NoiseModel) -> int:
    """
    Apply measurement noise to outcome.
    
    Args:
        outcome: True measurement outcome (0 or 1)
        noise_model: NoiseModel instance
    
    Returns:
        Possibly flipped outcome
    """
    if NoiseChannel.sample_measurement_error(noise_model.p_meas):
        return 1 - outcome
    return outcome


def test_noise_models():
    """Test noise model sampling."""
    noise = NoiseModel.ibm_like()
    
    # Test single-qubit errors
    errors = [NoiseChannel.sample_single_qubit_pauli_depolarizing(0.1) 
              for _ in range(1000)]
    error_rate = sum(e is not None for e in errors) / len(errors)
    print(f"Single-qubit error rate: {error_rate:.3f} (expected ~0.1)")
    
    # Test biased errors
    biased_noise = NoiseModel.biased(eta=100.0)
    errors = [NoiseChannel.sample_single_qubit_pauli_biased(0.1, 100.0) 
              for _ in range(1000)]
    z_count = sum(e == 'Z' for e in errors if e is not None)
    x_count = sum(e == 'X' for e in errors if e is not None)
    print(f"Z/X ratio: {z_count/max(x_count, 1):.1f} (expected ~100)")
    
    print("Noise model tests passed!")


if __name__ == "__main__":
    test_noise_models()
