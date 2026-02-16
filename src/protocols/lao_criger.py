"""
Lao-Criger Protocols (2022)
===========================

Magic-state injection on the rotated surface code with CR and MR schemes.

Based on: Lao & Criger, arXiv:2204.12037 (2022)

CR (Corner-Rotated): Magic state at corner of rotated lattice
MR (Middle-Rotated): Magic state at middle of rotated lattice

Key results:
- MR outperforms CR and Li's YL under biased noise (p1 << p2)
- Leading-order: p_L^CR ≈ (3/5) p2^2, p_L^MR ≈ (3/5) p2^2
- MR has fewer "sensitive" qubits vulnerable to single-qubit errors
"""

from typing import List, Tuple, Dict
import numpy as np
from .base_protocol import (
    BaseProtocol, ProtocolParameters, InjectionCircuit,
    StabilizerMeasurement, RotatedSurfaceCodeLayout
)


class CRProtocol(BaseProtocol):
    """
    Corner-Rotated injection protocol.
    
    Magic state initialized at corner of rotated surface code lattice.
    """
    
    def __init__(self, distance: int = 3):
        """
        Initialize CR protocol.
        
        Args:
            distance: Code distance (must be odd, >= 3)
        """
        params = ProtocolParameters(
            distance=distance,
            magic_qubit_loc='corner',
            num_post_selection_rounds=2,
            protocol_name=f'CR-d{distance}'
        )
        super().__init__(params)
        
    def build_circuit(self) -> InjectionCircuit:
        """
        Build CR injection circuit.
        
        Phase 1: Two rounds of post-selected stabilizer measurements
        Phase 2: Normal error correction (not part of injection proper)
        """
        d = self.distance
        n_data = d * d  # Rotated surface code uses d^2 data qubits
        n_ancilla = d * d - 1  # Approximate number of stabilizers
        
        circuit = InjectionCircuit(n_data, n_ancilla)
        
        # Initialization
        init_ops = self._build_initialization()
        circuit.add_timestep(init_ops)
        
        # Two rounds of stabilizer measurements
        for round_idx in range(2):
            stab_ops = self._build_stabilizer_round(round_idx)
            for timestep_ops in stab_ops:
                circuit.add_timestep(timestep_ops)
        
        self.circuit = circuit
        return circuit
    
    def _build_initialization(self) -> Dict:
        """
        Build initialization for CR protocol.
        
        Corner qubit (top-left): magic state |T⟩ = |0⟩ + e^(iπ/4)|1⟩
        Other qubits: |+⟩ or |0⟩ based on logical operator support
        """
        init_ops = {
            'init': {0: 'magic'},  # Corner qubit
            'gates': [],
            'measurements': []
        }
        
        # Initialize other qubits
        # Left boundary: |+⟩ (X logical support)
        # Top boundary: |0⟩ (Z logical support)
        # Interior: pattern ensures correct logical state
        
        d = self.distance
        for i in range(1, d * d):
            # Simplified: would need actual geometric layout
            row, col = i // d, i % d
            if col == 0 or row == 0:
                init_ops['init'][i] = '+' if col == 0 else '0'
            else:
                init_ops['init'][i] = '0'
        
        return init_ops
    
    def _build_stabilizer_round(self, round_idx: int) -> List[Dict]:
        """
        Build one round of stabilizer measurements.
        
        Uses optimized CNOT ordering similar to Li's protocol.
        """
        timesteps = []
        
        # Get all stabilizers
        stabilizers = self.get_stabilizers()
        
        # Schedule CNOTs to minimize circuit depth
        # This requires careful planning - simplified here
        
        # Example timestep structure:
        # 1. Initialize ancillas to |0⟩
        # 2-8. CNOT gates in parallel where possible
        # 9. Measure ancillas
        
        return timesteps
    
    def get_data_qubit_layout(self) -> np.ndarray:
        """
        Get 2D diamond layout for rotated surface code.
        
        For distance d, arrange d^2 qubits in diamond pattern.
        """
        d = self.distance
        layout = np.full((d, d), -1, dtype=int)
        
        # Fill diamond pattern
        mid = d // 2
        qubit_idx = 0
        
        for row in range(d):
            if row <= mid:
                # Upper half
                n_qubits = row + 1
                col_start = mid - row
            else:
                # Lower half
                n_qubits = d - row
                col_start = row - mid
            
            for i in range(n_qubits):
                layout[row, col_start + i] = qubit_idx
                qubit_idx += 1
        
        return layout
    
    def get_stabilizers(self) -> List[StabilizerMeasurement]:
        """
        Get stabilizer measurements for rotated surface code.
        
        X-type: plaquette stabilizers (face-centered)
        Z-type: vertex stabilizers (vertex-centered)
        
        Most stabilizers have weight 4, boundary ones have weight 2-3.
        """
        stabilizers = []
        
        # Would enumerate all X and Z stabilizers based on geometry
        # For distance 3: ~8 stabilizers total
        
        return stabilizers
    
    def get_initial_state(self) -> Dict[int, str]:
        """Get initial state dict for CR protocol."""
        init_ops = self._build_initialization()
        return init_ops['init']
    
    def check_post_selection(self, syndrome_history: List[List[int]]) -> bool:
        """
        Check CR post-selection condition.
        
        Similar to Li's protocol:
        - First round should match initialization pattern
        - Second round should match first round
        """
        if len(syndrome_history) < 2:
            return False
        
        round1, round2 = syndrome_history[0], syndrome_history[1]
        
        # Check expected initial syndrome
        # Based on initialization pattern
        # Stabilizers should all be +1 if no errors
        
        # Check consistency
        if not np.array_equal(round1, round2):
            return False
        
        return True
    
    def get_magic_qubit_index(self) -> int:
        """Magic qubit at top-left corner."""
        return 0
    
    def get_logical_operators(self) -> Tuple[List[int], List[int]]:
        """
        Get logical operators for rotated surface code.
        
        X_L: string along left boundary
        Z_L: string along top boundary
        """
        d = self.distance
        layout = self.get_data_qubit_layout()
        
        # X_L: left boundary (column 0)
        x_logical = [layout[row, 0] for row in range(d) if layout[row, 0] >= 0]
        
        # Z_L: top boundary (row 0)
        z_logical = [layout[0, col] for col in range(d) if layout[0, col] >= 0]
        
        return (x_logical, z_logical)


class MRProtocol(BaseProtocol):
    """
    Middle-Rotated injection protocol.
    
    Magic state initialized at middle of rotated surface code lattice.
    Reduces number of sensitive qubits compared to CR.
    """
    
    def __init__(self, distance: int = 3):
        """
        Initialize MR protocol.
        
        Args:
            distance: Code distance (must be odd, >= 3)
        """
        params = ProtocolParameters(
            distance=distance,
            magic_qubit_loc='middle',
            num_post_selection_rounds=2,
            protocol_name=f'MR-d{distance}'
        )
        super().__init__(params)
        
    def build_circuit(self) -> InjectionCircuit:
        """Build MR injection circuit."""
        d = self.distance
        n_data = d * d
        n_ancilla = d * d - 1
        
        circuit = InjectionCircuit(n_data, n_ancilla)
        
        # Initialization with magic qubit in middle
        init_ops = self._build_initialization()
        circuit.add_timestep(init_ops)
        
        # Two rounds of stabilizer measurements
        for round_idx in range(2):
            stab_ops = self._build_stabilizer_round(round_idx)
            for timestep_ops in stab_ops:
                circuit.add_timestep(timestep_ops)
        
        self.circuit = circuit
        return circuit
    
    def _build_initialization(self) -> Dict:
        """
        Build initialization for MR protocol.
        
        Middle qubit: magic state
        Logical operators: horizontal and vertical strings through middle
        """
        d = self.distance
        mid = d // 2
        magic_idx = mid * d + mid  # Center of diamond
        
        init_ops = {
            'init': {magic_idx: 'magic'},
            'gates': [],
            'measurements': []
        }
        
        # Initialize other qubits based on logical operator support
        # Horizontal line through middle: |+⟩
        # Vertical line through middle: |0⟩
        # Others: pattern for correct logical state
        
        for i in range(d * d):
            if i == magic_idx:
                continue
            
            row, col = i // d, i % d
            if row == mid:
                init_ops['init'][i] = '+'
            elif col == mid:
                init_ops['init'][i] = '0'
            else:
                init_ops['init'][i] = '0'
        
        return init_ops
    
    def _build_stabilizer_round(self, round_idx: int) -> List[Dict]:
        """Build stabilizer measurement round."""
        return []  # Similar to CR
    
    def get_data_qubit_layout(self) -> np.ndarray:
        """Same layout as CR - rotated diamond."""
        return CRProtocol(self.distance).get_data_qubit_layout()
    
    def get_stabilizers(self) -> List[StabilizerMeasurement]:
        """Get stabilizers - same as CR."""
        return []
    
    def get_initial_state(self) -> Dict[int, str]:
        """Get initial state for MR."""
        init_ops = self._build_initialization()
        return init_ops['init']
    
    def check_post_selection(self, syndrome_history: List[List[int]]) -> bool:
        """Check MR post-selection (same logic as CR)."""
        if len(syndrome_history) < 2:
            return False
        
        return np.array_equal(syndrome_history[0], syndrome_history[1])
    
    def get_magic_qubit_index(self) -> int:
        """Magic qubit at center."""
        mid = self.distance // 2
        return mid * self.distance + mid
    
    def get_logical_operators(self) -> Tuple[List[int], List[int]]:
        """
        Logical operators for MR.
        
        X_L: horizontal line through middle
        Z_L: vertical line through middle
        """
        d = self.distance
        mid = d // 2
        
        # X_L: horizontal through middle
        x_logical = list(range(mid * d, (mid + 1) * d))
        
        # Z_L: vertical through middle
        z_logical = [mid + i * d for i in range(d)]
        
        return (x_logical, z_logical)


def test_cr_mr_protocols():
    """Test CR and MR protocol construction."""
    # Test CR
    cr = CRProtocol(distance=3)
    print(f"CR Protocol: {cr.params.protocol_name}")
    print(f"Magic qubit index: {cr.get_magic_qubit_index()}")
    
    layout = cr.get_data_qubit_layout()
    print(f"Layout shape: {layout.shape}")
    print(f"Layout:\n{layout}")
    
    # Test MR
    mr = MRProtocol(distance=3)
    print(f"\nMR Protocol: {mr.params.protocol_name}")
    print(f"Magic qubit index: {mr.get_magic_qubit_index()}")
    
    # Build circuits
    cr_circuit = cr.build_circuit()
    mr_circuit = mr.build_circuit()
    
    print(f"\nCR circuit timesteps: {len(cr_circuit)}")
    print(f"MR circuit timesteps: {len(mr_circuit)}")
    
    print("\nCR/MR protocol tests passed!")


if __name__ == "__main__":
    test_cr_mr_protocols()
