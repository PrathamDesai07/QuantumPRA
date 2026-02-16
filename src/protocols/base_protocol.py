"""
Base Magic-State Injection Protocol
====================================

Abstract base class for magic-state injection protocols on surface codes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np


@dataclass
class ProtocolParameters:
    """
    Parameters defining an injection protocol.
    
    Attributes:
        distance: Code distance
        magic_qubit_loc: Location of magic qubit ('corner', 'middle', 'edge')
        num_post_selection_rounds: Number of stabilizer measurement rounds for post-selection
        protocol_name: Human-readable protocol name
    """
    distance: int
    magic_qubit_loc: str
    num_post_selection_rounds: int
    protocol_name: str
    
    def __post_init__(self):
        """Validate parameters."""
        assert self.distance >= 3 and self.distance % 2 == 1, "Distance must be odd and >= 3"
        assert self.magic_qubit_loc in ['corner', 'middle', 'edge']
        assert self.num_post_selection_rounds >= 1


@dataclass
class StabilizerMeasurement:
    """
    Represents a single stabilizer measurement.
    
    Attributes:
        type: 'X' or 'Z'
        qubits: List of data qubit indices
        ancilla: Ancilla qubit index
        cnot_order: Order of CNOT gates [(control, target), ...]
    """
    type: str  # 'X' or 'Z'
    qubits: List[int]
    ancilla: int
    cnot_order: List[Tuple[int, int]]
    
    def __post_init__(self):
        assert self.type in ['X', 'Z']
        assert len(self.qubits) in [3, 4], "Surface code stabilizers have weight 3 or 4"


class InjectionCircuit:
    """
    Represents the full injection circuit with timesteps.
    
    A circuit is organized into timesteps, where each timestep contains
    parallel operations that can be executed simultaneously.
    """
    
    def __init__(self, n_data_qubits: int, n_ancilla_qubits: int):
        self.n_data = n_data_qubits
        self.n_ancilla = n_ancilla_qubits
        self.n_total = n_data_qubits + n_ancilla_qubits
        self.timesteps: List[Dict] = []
        
    def add_timestep(self, operations: Dict):
        """
        Add a timestep with operations.
        
        Args:
            operations: Dict with keys 'init', 'gates', 'measurements'
        """
        self.timesteps.append(operations)
    
    def __len__(self):
        return len(self.timesteps)


class BaseProtocol(ABC):
    """
    Abstract base class for magic-state injection protocols.
    """
    
    def __init__(self, params: ProtocolParameters):
        """
        Initialize protocol.
        
        Args:
            params: Protocol parameters
        """
        self.params = params
        self.distance = params.distance
        self.circuit: Optional[InjectionCircuit] = None
        
    @abstractmethod
    def build_circuit(self) -> InjectionCircuit:
        """
        Build the injection circuit.
        
        Returns:
            InjectionCircuit instance
        """
        pass
    
    @abstractmethod
    def get_data_qubit_layout(self) -> np.ndarray:
        """
        Get 2D layout of data qubits.
        
        Returns:
            2D array with qubit indices (-1 for empty positions)
        """
        pass
    
    @abstractmethod
    def get_stabilizers(self) -> List[StabilizerMeasurement]:
        """
        Get list of stabilizer measurements.
        
        Returns:
            List of StabilizerMeasurement instances
        """
        pass
    
    @abstractmethod
    def get_initial_state(self) -> Dict[int, str]:
        """
        Get initial state for each data qubit.
        
        Returns:
            Dict mapping qubit index to initial state ('0', '+', 'magic')
        """
        pass
    
    @abstractmethod
    def check_post_selection(self, syndrome_history: List[List[int]]) -> bool:
        """
        Check if post-selection condition is satisfied.
        
        Args:
            syndrome_history: List of syndrome measurements for each round
        
        Returns:
            True if run should be accepted, False if rejected
        """
        pass
    
    def get_magic_qubit_index(self) -> int:
        """
        Get the data qubit index where magic state is initialized.
        
        Returns:
            Qubit index
        """
        raise NotImplementedError("Subclass must implement")
    
    def get_logical_operators(self) -> Tuple[List[int], List[int]]:
        """
        Get logical X and Z operators.
        
        Returns:
            Tuple of (X_L qubits, Z_L qubits)
        """
        raise NotImplementedError("Subclass must implement")


class RotatedSurfaceCodeLayout:
    """
    Helper class for rotated surface code geometry.
    """
    
    @staticmethod
    def get_data_qubits(distance: int) -> List[Tuple[int, int]]:
        """
        Get data qubit positions for rotated surface code.
        
        For distance d, we have d^2 data qubits arranged in a diamond.
        
        Args:
            distance: Code distance (must be odd)
        
        Returns:
            List of (row, col) positions
        """
        assert distance % 2 == 1
        d = distance
        qubits = []
        
        # Diamond shape layout
        mid = d // 2
        for row in range(d):
            if row <= mid:
                # Upper half: increasing width
                width = row + 1
                col_start = mid - row // 2
            else:
                # Lower half: decreasing width
                width = d - row
                col_start = row // 2 - mid + (d - row - 1) // 2
            
            for i in range(width):
                qubits.append((row, col_start + i))
        
        return qubits
    
    @staticmethod
    def get_x_stabilizers(distance: int) -> List[List[Tuple[int, int]]]:
        """
        Get X-type stabilizer locations for rotated surface code.
        
        Returns:
            List of stabilizers, each is a list of (row, col) positions
        """
        # X stabilizers are centered on faces
        # Implementation depends on specific layout convention
        raise NotImplementedError("Implement based on chosen convention")
    
    @staticmethod
    def get_z_stabilizers(distance: int) -> List[List[Tuple[int, int]]]:
        """
        Get Z-type stabilizer locations for rotated surface code.
        
        Returns:
            List of stabilizers, each is a list of (row, col) positions
        """
        # Z stabilizers are centered on vertices
        raise NotImplementedError("Implement based on chosen convention")


def test_base_protocol():
    """Test base protocol utilities."""
    # Test layout generation
    d = 3
    qubits = RotatedSurfaceCodeLayout.get_data_qubits(d)
    assert len(qubits) == d * d, f"Expected {d*d} qubits, got {len(qubits)}"
    
    print(f"Generated {len(qubits)} data qubits for distance-{d} rotated surface code")
    print("Base protocol tests passed!")


if __name__ == "__main__":
    test_base_protocol()
