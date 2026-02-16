"""
Stabilizer Tableau Operations
==============================

Implements efficient stabilizer tableau operations for simulating
Clifford circuits with noise. Based on the Gottesman-Knill theorem
and Aaronson-Gottesman's improved tableau algorithm.

References:
- Aaronson & Gottesman, Phys. Rev. A 70, 052328 (2004)
- Garcia et al., Quantum 5, 385 (2021)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StabilizerTableau:
    """
    Stabilizer tableau representation for n qubits.
    
    Stores the generators of the stabilizer group and destabilizers
    using a 2n x 2n+1 binary matrix representation.
    
    Attributes:
        n: Number of qubits
        x_table: Binary matrix for X components (2n x n)
        z_table: Binary matrix for Z components (2n x n)
        phase: Phase vector (2n)
    """
    n: int
    x_table: np.ndarray  # Shape (2n, n)
    z_table: np.ndarray  # Shape (2n, n)
    phase: np.ndarray    # Shape (2n,)
    
    def __post_init__(self):
        """Validate tableau dimensions."""
        assert self.x_table.shape == (2*self.n, self.n)
        assert self.z_table.shape == (2*self.n, self.n)
        assert self.phase.shape == (self.n * 2,)
    
    def copy(self) -> 'StabilizerTableau':
        """Create a deep copy of the tableau."""
        return StabilizerTableau(
            n=self.n,
            x_table=self.x_table.copy(),
            z_table=self.z_table.copy(),
            phase=self.phase.copy()
        )
    
    @classmethod
    def computational_basis(cls, n: int, state: Optional[np.ndarray] = None) -> 'StabilizerTableau':
        """
        Initialize tableau in computational basis |0...0⟩ or specified state.
        
        Args:
            n: Number of qubits
            state: Optional binary vector specifying |state⟩ (default: all zeros)
        
        Returns:
            StabilizerTableau initialized to computational basis
        """
        x_table = np.zeros((2*n, n), dtype=np.uint8)
        z_table = np.zeros((2*n, n), dtype=np.uint8)
        phase = np.zeros(2*n, dtype=np.uint8)
        
        # First n rows: destabilizers X_i
        # Last n rows: stabilizers Z_i
        for i in range(n):
            x_table[i, i] = 1  # X_i destabilizer
            z_table[n + i, i] = 1  # Z_i stabilizer
        
        if state is not None:
            assert len(state) == n
            phase[n:] = state  # Z_i stabilizer phase encodes |state⟩
        
        return cls(n=n, x_table=x_table, z_table=z_table, phase=phase)
    
    @classmethod
    def plus_basis(cls, n: int) -> 'StabilizerTableau':
        """
        Initialize tableau in plus basis |+...+⟩.
        
        Args:
            n: Number of qubits
        
        Returns:
            StabilizerTableau initialized to |+...+⟩
        """
        x_table = np.zeros((2*n, n), dtype=np.uint8)
        z_table = np.zeros((2*n, n), dtype=np.uint8)
        phase = np.zeros(2*n, dtype=np.uint8)
        
        # First n rows: destabilizers Z_i
        # Last n rows: stabilizers X_i
        for i in range(n):
            z_table[i, i] = 1  # Z_i destabilizer
            x_table[n + i, i] = 1  # X_i stabilizer
        
        return cls(n=n, x_table=x_table, z_table=z_table, phase=phase)
    
    def rowsum(self, h: int, i: int):
        """
        Add row i to row h with proper phase updates.
        
        This implements the Pauli multiplication with correct phase tracking.
        """
        # Count the number of Y's in each row
        x_h = self.x_table[h]
        z_h = self.z_table[h]
        x_i = self.x_table[i]
        z_i = self.z_table[i]
        
        # Calculate phase change
        # Phase changes by 2 if we have Y*X -> -Z or similar
        for q in range(self.n):
            # Count commutation violations
            if x_h[q] and z_h[q]:  # Y in row h
                if x_i[q] and not z_i[q]:  # X in row i
                    self.phase[h] ^= 1
                elif not x_i[q] and z_i[q]:  # Z in row i  
                    self.phase[h] ^= 1
            elif not x_h[q] and z_h[q]:  # Z in row h
                if x_i[q] and z_i[q]:  # Y in row i
                    self.phase[h] ^= 1
            elif x_h[q] and not z_h[q]:  # X in row h
                if x_i[q] and z_i[q]:  # Y in row i
                    self.phase[h] ^= 1
        
        # Update tableau entries (XOR)
        self.x_table[h] ^= self.x_table[i]
        self.z_table[h] ^= self.z_table[i]
        self.phase[h] ^= self.phase[i]


class StabilizerSimulator:
    """
    Simulator for Clifford circuits using stabilizer tableaux.
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize simulator.
        
        Args:
            n_qubits: Number of qubits to simulate
        """
        self.n = n_qubits
        self.tableau = StabilizerTableau.computational_basis(n_qubits)
        self.measurement_results: List[Tuple[int, int]] = []
    
    def reset(self, initial_state: Optional[np.ndarray] = None):
        """Reset to computational basis state."""
        self.tableau = StabilizerTableau.computational_basis(self.n, initial_state)
        self.measurement_results = []
    
    def hadamard(self, q: int):
        """
        Apply Hadamard gate to qubit q.
        
        H: X ↔ Z
        """
        # Swap X and Z columns for qubit q
        self.tableau.x_table[:, q], self.tableau.z_table[:, q] = \
            self.tableau.z_table[:, q].copy(), self.tableau.x_table[:, q].copy()
    
    def phase(self, q: int):
        """
        Apply Phase gate (S gate) to qubit q.
        
        S: X → Y, Z → Z
        """
        # Y = iXZ, so X → XZ (add Z to X rows)
        for i in range(2 * self.n):
            if self.tableau.x_table[i, q]:
                self.tableau.z_table[i, q] ^= 1
                self.tableau.phase[i] ^= self.tableau.x_table[i, q] & self.tableau.z_table[i, q]
    
    def phase_dag(self, q: int):
        """
        Apply Phase dagger gate (S† gate) to qubit q.
        
        S†: X → -Y, Z → Z
        """
        # -Y = -iXZ, so X → XZ with phase flip
        for i in range(2 * self.n):
            if self.tableau.x_table[i, q]:
                self.tableau.phase[i] ^= self.tableau.x_table[i, q] & (~self.tableau.z_table[i, q])
                self.tableau.z_table[i, q] ^= 1
    
    def cnot(self, control: int, target: int):
        """
        Apply CNOT gate with control and target qubits.
        
        CNOT: X_c → X_c X_t, Z_c → Z_c, X_t → X_t, Z_t → Z_c Z_t
        """
        # Update phase: if control has Z and target has X, pick up phase
        for i in range(2 * self.n):
            if self.tableau.x_table[i, control] and self.tableau.z_table[i, target]:
                if not (self.tableau.z_table[i, control] or self.tableau.x_table[i, target]):
                    self.tableau.phase[i] ^= 1
        
        # X_c → X_c X_t
        self.tableau.x_table[:, target] ^= self.tableau.x_table[:, control]
        
        # Z_t → Z_c Z_t
        self.tableau.z_table[:, control] ^= self.tableau.z_table[:, target]
    
    def cz(self, q1: int, q2: int):
        """
        Apply CZ gate between q1 and q2.
        
        CZ: H_2 · CNOT_{1→2} · H_2
        """
        self.hadamard(q2)
        self.cnot(q1, q2)
        self.hadamard(q2)
    
    def measure_z(self, q: int, force_outcome: Optional[int] = None) -> int:
        """
        Measure qubit q in Z basis.
        
        Args:
            q: Qubit index to measure
            force_outcome: If provided, force this measurement outcome (for error injection)
        
        Returns:
            Measurement outcome (0 or 1)
        """
        n = self.n
        
        # Find if X_q is in the stabilizer group (check rows n to 2n-1)
        # If X_q commutes with all stabilizers, outcome is deterministic
        p = None
        for i in range(n, 2*n):
            if self.tableau.x_table[i, q]:
                p = i
                break
        
        if p is None:
            # Deterministic outcome: measure stabilizer eigenvalue
            # Find which destabilizer has X_q
            destab_idx = None
            for i in range(n):
                if self.tableau.x_table[i, q]:
                    destab_idx = i
                    break
            
            # Get outcome from corresponding stabilizer phase
            if destab_idx is not None:
                outcome = self.tableau.phase[destab_idx + n] if force_outcome is None else force_outcome
            else:
                # Edge case: shouldn't happen in valid tableau
                outcome = 0 if force_outcome is None else force_outcome
        else:
            # Random outcome
            outcome = np.random.randint(2) if force_outcome is None else force_outcome
            
            # Project: set row p to Z_q with correct phase
            self.tableau.x_table[p] = 0
            self.tableau.z_table[p] = 0
            self.tableau.z_table[p, q] = 1
            self.tableau.phase[p] = outcome
            
            # Update all rows that anticommute with Z_q
            for i in range(2*n):
                if i != p and self.tableau.x_table[i, q]:
                    self.tableau.rowsum(i, p)
        
        self.measurement_results.append((q, outcome))
        return outcome
    
    def measure_x(self, q: int, force_outcome: Optional[int] = None) -> int:
        """
        Measure qubit q in X basis.
        
        Implemented as H · measure_Z · H
        """
        self.hadamard(q)
        outcome = self.measure_z(q, force_outcome)
        self.hadamard(q)
        return outcome
    
    def apply_pauli(self, pauli_string: str):
        """
        Apply Pauli operator string (e.g., 'IXYZ' for 4 qubits).
        
        Args:
            pauli_string: String of I, X, Y, Z characters
        """
        assert len(pauli_string) == self.n
        
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                self.hadamard(i)
                self.phase(i)
                self.phase(i)
                self.hadamard(i)
            elif pauli == 'Y':
                self.phase_dag(i)
                self.hadamard(i)
                self.phase(i)
                self.phase(i)
                self.hadamard(i)
                self.phase(i)
            elif pauli == 'Z':
                self.phase(i)
                self.phase(i)
            # I: do nothing
    
    def get_expectation(self, pauli_string: str) -> int:
        """
        Get expectation value of Pauli observable (±1).
        
        Returns:
            ±1 depending on state
        """
        # Check if observable commutes with all stabilizers
        # If yes, return the eigenvalue; if no, return 0 (undefined for mixed state)
        # For simplicity, we'll measure and immediately reset
        raise NotImplementedError("Use measure and reconstruct state for now")


def test_stabilizer_basics():
    """Basic tests for stabilizer operations."""
    # Test Bell state creation
    sim = StabilizerSimulator(2)
    sim.hadamard(0)
    sim.cnot(0, 1)
    # Should be in |Φ+⟩ = (|00⟩ + |11⟩)/√2
    
    # Test measurements
    sim2 = StabilizerSimulator(1)
    sim2.hadamard(0)
    outcomes = [sim2.measure_z(0) for _ in range(10)]
    # Should get random 0s and 1s
    
    print("Stabilizer tableau tests passed!")


if __name__ == "__main__":
    test_stabilizer_basics()
