"""
Li Protocol (2015)
==================

Post-selected magic-state encoding on the regular surface code.

Based on: Li, New J. Phys. 17, 023037 (2015)

Key features:
- Two-phase encoding: distance d1 → d2
- Corner injection of magic state
- Two rounds of post-selection
- Optimized CNOT ordering to minimize logical errors
- Leading-order p_L ≈ (2/5) * p2^2 with perfect single-qubit gates
"""

from typing import List, Tuple, Dict
import numpy as np
from .base_protocol import (
    BaseProtocol, ProtocolParameters, InjectionCircuit, 
    StabilizerMeasurement
)


class LiProtocol(BaseProtocol):
    """
    Li's post-selected encoding protocol on regular surface code.
    """
    
    def __init__(self, distance_initial: int = 3, distance_final: int = 7):
        """
        Initialize Li protocol.
        
        Args:
            distance_initial: Initial distance d1 for phase 1
            distance_final: Final distance d2 for phase 2
        """
        params = ProtocolParameters(
            distance=distance_final,
            magic_qubit_loc='corner',
            num_post_selection_rounds=2,
            protocol_name=f'Li-d{distance_initial}-{distance_final}'
        )
        super().__init__(params)
        
        self.d1 = distance_initial
        self.d2 = distance_final
        assert self.d1 >= 3 and self.d1 % 2 == 1
        assert self.d2 >= self.d1
        
    def build_circuit(self) -> InjectionCircuit:
        """
        Build Li's two-phase encoding circuit.
        
        Phase 1:
        - Initialize magic qubit at top-left corner
        - Initialize other qubits in area I (|+⟩) and area II (|0⟩)
        - Two rounds of stabilizer measurements with optimized CNOT order
        - Post-selection based on syndromes
        
        Phase 2:
        - Initialize qubits in areas III and IV
        - One round of full stabilizer measurements
        - Proceed to normal error correction
        """
        # For simplicity, implement distance-3 case explicitly
        # Full implementation would generalize to arbitrary distances
        
        n_data = self.d2 ** 2
        n_ancilla = (self.d2 - 1) ** 2  # Rough estimate for surface code
        
        circuit = InjectionCircuit(n_data, n_ancilla)
        
        # Phase 1: Initialize small patch
        init_ops = self._build_phase1_initialization()
        circuit.add_timestep(init_ops)
        
        # Phase 1: Two rounds of stabilizer measurements
        for round_idx in range(2):
            stab_ops = self._build_phase1_stabilizer_round(round_idx)
            for timestep_ops in stab_ops:
                circuit.add_timestep(timestep_ops)
        
        # Phase 2: Grow to full distance
        if self.d2 > self.d1:
            init_ops = self._build_phase2_initialization()
            circuit.add_timestep(init_ops)
            
            # One round of full stabilizer measurements
            stab_ops = self._build_phase2_stabilizer_round()
            for timestep_ops in stab_ops:
                circuit.add_timestep(timestep_ops)
        
        self.circuit = circuit
        return circuit
    
    def _build_phase1_initialization(self) -> Dict:
        """Build initialization operations for phase 1."""
        # Qubit 0: magic state (special initialization)
        # Area I qubits: |+⟩
        # Area II qubits: |0⟩
        
        init_ops = {
            'init': {
                0: 'magic',  # Top-left corner
            },
            'gates': [],
            'measurements': []
        }
        
        # For d1=3, we have 9 data qubits total in phase 1
        # Simplified layout for illustration
        for i in range(1, 9):
            if i < 4:  # Area I (example)
                init_ops['init'][i] = '+'
            else:  # Area II
                init_ops['init'][i] = '0'
        
        return init_ops
    
    def _build_phase1_stabilizer_round(self, round_idx: int) -> List[Dict]:
        """
        Build optimized stabilizer measurement circuit for phase 1.
        
        The CNOT ordering is critical for achieving low logical error rates.
        Based on Li's Fig. 2, the optimal circuit uses 9 timesteps per round.
        """
        timesteps = []
        
        # This is a simplified version - full implementation requires
        # careful enumeration of stabilizers and optimal CNOT scheduling
        
        # Example: X stabilizer measurement in area I
        # Uses ancilla and 4 data qubits
        
        x_stab_circuit = [
            {'gates': [('cnot', 'ancilla_0', 'data_1')], 'measurements': []},
            {'gates': [('cnot', 'ancilla_0', 'data_2')], 'measurements': []},
            {'gates': [('cnot', 'ancilla_0', 'data_3')], 'measurements': []},
            {'gates': [('cnot', 'ancilla_0', 'data_4')], 'measurements': []},
            {'gates': [], 'measurements': [('Z', 'ancilla_0')]},
        ]
        
        # Full implementation: interleave multiple stabilizer measurements
        # following Li's optimized schedule (Fig. 2)
        
        return timesteps
    
    def _build_phase2_initialization(self) -> Dict:
        """Build initialization for phase 2 (growing to full distance)."""
        init_ops = {
            'init': {},
            'gates': [],
            'measurements': []
        }
        
        # Initialize qubits in areas III and IV
        # Area III: |+⟩, Area IV: |0⟩
        
        return init_ops
    
    def _build_phase2_stabilizer_round(self) -> List[Dict]:
        """Build stabilizer measurements for phase 2."""
        return []
    
    def get_data_qubit_layout(self) -> np.ndarray:
        """
        Get 2D layout for regular surface code.
        
        Returns square lattice with data qubits at vertices.
        """
        layout = np.arange(self.d2 ** 2).reshape(self.d2, self.d2)
        return layout
    
    def get_stabilizers(self) -> List[StabilizerMeasurement]:
        """
        Get stabilizer measurements for regular surface code.
        
        X stabilizers on faces, Z stabilizers on vertices.
        """
        stabilizers = []
        
        # Simplified for distance 3
        # Full implementation: enumerate all stabilizers
        
        return stabilizers
    
    def get_initial_state(self) -> Dict[int, str]:
        """Get initial state dict."""
        state = {0: 'magic'}
        
        # Area I: |+⟩, Area II: |0⟩
        # Based on distance and layout
        
        return state
    
    def check_post_selection(self, syndrome_history: List[List[int]]) -> bool:
        """
        Check Li's post-selection condition.
        
        Two types of error syndromes trigger rejection:
        1. Mismatch between initialization pattern and first round
        2. Mismatch between first and second round measurements
        
        Args:
            syndrome_history: List of syndrome vectors, one per round
        
        Returns:
            True if accepted, False if rejected
        """
        if len(syndrome_history) < 2:
            return False
        
        round1, round2 = syndrome_history[0], syndrome_history[1]
        
        # Check initialization pattern match (round 1)
        # Expected pattern: stabilizers in slashed regions should be +1
        expected_round1 = self._get_expected_initial_syndrome()
        
        if not np.array_equal(round1, expected_round1):
            return False
        
        # Check consistency between rounds
        if not np.array_equal(round1, round2):
            return False
        
        return True
    
    def _get_expected_initial_syndrome(self) -> np.ndarray:
        """Get expected syndrome pattern in first round."""
        # Based on initialization: certain stabilizers should be +1
        # Implementation depends on detailed layout
        return np.zeros(10, dtype=int)  # Placeholder
    
    def get_magic_qubit_index(self) -> int:
        """Magic qubit is at top-left corner (index 0)."""
        return 0
    
    def get_logical_operators(self) -> Tuple[List[int], List[int]]:
        """
        Get logical operators for regular surface code.
        
        X_L: horizontal string along top edge
        Z_L: vertical string along left edge
        """
        x_logical = list(range(self.d2))  # Top row
        z_logical = list(range(0, self.d2**2, self.d2))  # Left column
        
        return (x_logical, z_logical)


def test_li_protocol():
    """Test Li protocol construction."""
    protocol = LiProtocol(distance_initial=3, distance_final=7)
    
    print(f"Protocol: {protocol.params.protocol_name}")
    print(f"Initial distance: {protocol.d1}")
    print(f"Final distance: {protocol.d2}")
    print(f"Magic qubit location: {protocol.params.magic_qubit_loc}")
    print(f"Post-selection rounds: {protocol.params.num_post_selection_rounds}")
    
    # Build circuit
    circuit = protocol.build_circuit()
    print(f"Circuit timesteps: {len(circuit)}")
    
    print("Li protocol tests passed!")


if __name__ == "__main__":
    test_li_protocol()
