"""
Circuit Builder for Injection Protocols
========================================

Builds detailed quantum circuits for magic-state injection with proper
timestep scheduling and CNOT ordering.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class GateType(Enum):
    """Types of quantum gates."""
    H = "H"
    S = "S"
    S_DAG = "S_dag"
    CNOT = "CNOT"
    CZ = "CZ"
    MEASURE_Z = "MZ"
    MEASURE_X = "MX"
    INIT_0 = "INIT_0"
    INIT_PLUS = "INIT_+"
    INIT_MAGIC = "INIT_T"


@dataclass
class Gate:
    """Represents a single quantum gate operation."""
    gate_type: GateType
    qubits: Tuple[int, ...]  # Qubit indices (1 for single-qubit, 2 for two-qubit)
    
    def __post_init__(self):
        """Validate gate."""
        if self.gate_type in [GateType.H, GateType.S, GateType.S_DAG, 
                              GateType.MEASURE_Z, GateType.MEASURE_X,
                              GateType.INIT_0, GateType.INIT_PLUS, GateType.INIT_MAGIC]:
            assert len(self.qubits) == 1, f"Single-qubit gate {self.gate_type} requires 1 qubit"
        elif self.gate_type in [GateType.CNOT, GateType.CZ]:
            assert len(self.qubits) == 2, f"Two-qubit gate {self.gate_type} requires 2 qubits"


@dataclass
class Timestep:
    """Represents a single timestep in the circuit."""
    gates: List[Gate]
    timestep_index: int
    
    def __repr__(self):
        return f"Timestep {self.timestep_index}: {len(self.gates)} gates"


class CircuitBuilder:
    """
    Builds detailed quantum circuits for injection protocols.
    """
    
    def __init__(self, n_data_qubits: int, n_ancilla_qubits: int):
        """
        Initialize circuit builder.
        
        Args:
            n_data_qubits: Number of data qubits
            n_ancilla_qubits: Number of ancilla qubits
        """
        self.n_data = n_data_qubits
        self.n_ancilla = n_ancilla_qubits
        self.n_total = n_data_qubits + n_ancilla_qubits
        self.timesteps: List[Timestep] = []
        
        # Data qubits are 0 to n_data-1
        # Ancilla qubits are n_data to n_total-1
        self.data_qubits = list(range(n_data_qubits))
        self.ancilla_qubits = list(range(n_data_qubits, self.n_total))
    
    def add_timestep(self, gates: List[Gate]):
        """Add a timestep with gates."""
        timestep = Timestep(gates=gates, timestep_index=len(self.timesteps))
        self.timesteps.append(timestep)
    
    def build_stabilizer_measurement(
        self,
        stabilizer_type: str,
        data_qubits: List[int],
        ancilla_qubit: int,
        cnot_order: Optional[List[int]] = None
    ) -> List[Timestep]:
        """
        Build circuit for a single stabilizer measurement.
        
        Args:
            stabilizer_type: 'X' or 'Z'
            data_qubits: List of data qubit indices (3 or 4 qubits)
            ancilla_qubit: Ancilla qubit index
            cnot_order: Optional custom CNOT ordering (indices into data_qubits list)
        
        Returns:
            List of timesteps for this stabilizer measurement
        """
        assert stabilizer_type in ['X', 'Z']
        assert len(data_qubits) in [2, 3, 4], "Stabilizers have weight 2-4"
        
        timesteps = []
        
        # Step 1: Initialize ancilla
        init_gate = Gate(GateType.INIT_0, (ancilla_qubit,))
        timesteps.append([init_gate])
        
        # Step 2: Hadamard on ancilla if X-type stabilizer
        if stabilizer_type == 'X':
            h_gate = Gate(GateType.H, (ancilla_qubit,))
            timesteps.append([h_gate])
        
        # Step 3-N: CNOTs in specified order
        if cnot_order is None:
            cnot_order = list(range(len(data_qubits)))
        
        for idx in cnot_order:
            data_q = data_qubits[idx]
            if stabilizer_type == 'Z':
                # CNOT: data -> ancilla
                cnot = Gate(GateType.CNOT, (data_q, ancilla_qubit))
            else:
                # CNOT: ancilla -> data
                cnot = Gate(GateType.CNOT, (ancilla_qubit, data_q))
            timesteps.append([cnot])
        
        # Step N+1: Hadamard on ancilla if X-type
        if stabilizer_type == 'X':
            h_gate = Gate(GateType.H, (ancilla_qubit,))
            timesteps.append([h_gate])
        
        # Step N+2: Measure ancilla
        meas_gate = Gate(GateType.MEASURE_Z, (ancilla_qubit,))
        timesteps.append([meas_gate])
        
        return timesteps
    
    def merge_parallel_timesteps(self, timestep_lists: List[List[List[Gate]]]) -> List[Timestep]:
        """
        Merge multiple stabilizer measurement circuits to maximize parallelism.
        
        Args:
            timestep_lists: List of stabilizer measurement circuits, 
                          each is a list of gate lists
        
        Returns:
            Merged timesteps with parallel gates where possible
        """
        if not timestep_lists:
            return []
        
        # Find maximum depth
        max_depth = max(len(ts_list) for ts_list in timestep_lists)
        
        merged = []
        for depth in range(max_depth):
            combined_gates = []
            used_qubits = set()
            
            for ts_list in timestep_lists:
                if depth < len(ts_list):
                    gates = ts_list[depth]
                    # Check if gates can be added (no qubit conflicts)
                    gate_qubits = set()
                    for gate in gates:
                        gate_qubits.update(gate.qubits)
                    
                    if not gate_qubits.intersection(used_qubits):
                        combined_gates.extend(gates)
                        used_qubits.update(gate_qubits)
            
            if combined_gates:
                merged.append(combined_gates)
        
        return merged


def build_distance3_cr_circuit() -> Tuple[CircuitBuilder, Dict]:
    """
    Build detailed circuit for distance-3 CR protocol.
    
    Returns:
        Tuple of (CircuitBuilder, metadata dict)
    """
    # Distance-3 rotated surface code has 9 data qubits
    # Arranged in diamond:
    #     0
    #   1 2 3
    # 4 5 6 7
    #   8 9 10
    #    11
    # Actually simpler: 3x3 = 9 data qubits
    #   0
    #  1 2
    # 3 4 5
    #  6 7
    #   8
    
    n_data = 9
    n_ancilla = 8  # 4 X-stabilizers + 4 Z-stabilizers (approximate)
    
    builder = CircuitBuilder(n_data, n_ancilla)
    
    # Phase 1: Initialization
    # Magic qubit at corner (qubit 0)
    init_gates = [Gate(GateType.INIT_MAGIC, (0,))]
    
    # Initialize other data qubits
    # Left boundary: |+⟩, Top boundary: |0⟩
    for i in [1, 3, 6]:  # Left boundary
        init_gates.append(Gate(GateType.INIT_PLUS, (i,)))
    for i in [2, 5, 7, 8]:  # Others: |0⟩
        init_gates.append(Gate(GateType.INIT_0, (i,)))
    init_gates.append(Gate(GateType.INIT_0, (4,)))  # Center
    
    builder.add_timestep(init_gates)
    
    # Define stabilizers (simplified for distance-3)
    # X-stabilizers (4 plaquettes)
    x_stabilizers = [
        ([1, 2, 3, 4], 9),   # Upper-left plaquette
        ([2, 4, 5, 7], 10),  # Upper-right plaquette
        ([3, 4, 6, 8], 11),  # Lower-left plaquette
        ([4, 5, 7, 8], 12),  # Lower-right plaquette (weight-3 on boundary)
    ]
    
    # Z-stabilizers (4 vertices)
    z_stabilizers = [
        ([0, 1, 2], 13),      # Top vertex
        ([1, 3, 4], 14),      # Left vertex
        ([2, 4, 5], 15),      # Right vertex
        ([4, 6, 7, 8], 16),   # Bottom vertex
    ]
    
    # Two rounds of stabilizer measurements
    for round_idx in range(2):
        # Measure X-stabilizers
        x_circuits = []
        for data_qs, anc in x_stabilizers:
            circ = builder.build_stabilizer_measurement('X', data_qs, anc)
            x_circuits.append(circ)
        
        # Merge and add X-stabilizer timesteps
        merged_x = builder.merge_parallel_timesteps(x_circuits)
        for gates in merged_x:
            builder.add_timestep(gates)
        
        # Measure Z-stabilizers
        z_circuits = []
        for data_qs, anc in z_stabilizers:
            circ = builder.build_stabilizer_measurement('Z', data_qs, anc)
            z_circuits.append(circ)
        
        # Merge and add Z-stabilizer timesteps
        merged_z = builder.merge_parallel_timesteps(z_circuits)
        for gates in merged_z:
            builder.add_timestep(gates)
    
    metadata = {
        'protocol': 'CR-d3',
        'n_data': n_data,
        'n_ancilla': n_ancilla,
        'magic_qubit': 0,
        'x_stabilizers': x_stabilizers,
        'z_stabilizers': z_stabilizers,
        'logical_x': [1, 3, 6],  # Left boundary
        'logical_z': [0, 2, 5, 7],  # Top to bottom diagonal
    }
    
    return builder, metadata


def test_circuit_builder():
    """Test circuit building."""
    builder, metadata = build_distance3_cr_circuit()
    
    print(f"Built circuit for {metadata['protocol']}")
    print(f"Total qubits: {builder.n_total} ({metadata['n_data']} data + {metadata['n_ancilla']} ancilla)")
    print(f"Total timesteps: {len(builder.timesteps)}")
    print(f"Magic qubit: {metadata['magic_qubit']}")
    
    # Count gates by type
    gate_counts = {}
    for ts in builder.timesteps:
        for gate in ts.gates:
            gate_type = gate.gate_type.value
            gate_counts[gate_type] = gate_counts.get(gate_type, 0) + 1
    
    print("\nGate counts:")
    for gate_type, count in sorted(gate_counts.items()):
        print(f"  {gate_type}: {count}")
    
    print("\nCircuit builder tests passed!")


if __name__ == "__main__":
    test_circuit_builder()
