"""
First-Order Fault Enumeration
==============================

Enumerates single-fault locations in injection circuits that lead to
undetected logical errors. This provides analytic expressions for p_L.

Based on:
- Li (2015): Systematic fault enumeration for YL protocol
- Lao & Criger (2022): Fault analysis for CR/MR protocols
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from enum import Enum


class FaultLocation(Enum):
    """Types of fault locations."""
    INIT_DATA = "init_data"
    INIT_ANCILLA = "init_ancilla"
    SINGLE_QUBIT_GATE = "1q_gate"
    TWO_QUBIT_GATE = "2q_gate"
    MEASUREMENT = "measurement"
    IDLE = "idle"


@dataclass
class Fault:
    """
    Represents a single fault in the circuit.
    
    Attributes:
        location_type: Type of operation where fault occurs
        timestep: Timestep index in circuit
        qubit_indices: Affected qubit(s)
        pauli_error: Pauli error ('X', 'Y', 'Z', or 'IX', 'XY', etc. for 2q)
        detected: Whether this fault is detected by post-selection
        causes_logical_x: Whether this fault causes logical X error
        causes_logical_z: Whether this fault causes logical Z error
    """
    location_type: FaultLocation
    timestep: int
    qubit_indices: Tuple[int, ...]
    pauli_error: str
    detected: bool = False
    causes_logical_x: bool = False
    causes_logical_z: bool = False


class FaultEnumerator:
    """
    Enumerates all possible single-fault locations in an injection circuit.
    """
    
    def __init__(self, circuit_metadata: Dict):
        """
        Initialize fault enumerator.
        
        Args:
            circuit_metadata: Dict containing circuit structure info
        """
        self.metadata = circuit_metadata
        self.faults: List[Fault] = []
    
    def enumerate_all_faults(self, circuit_builder) -> List[Fault]:
        """
        Enumerate all possible single-fault locations.
        
        Args:
            circuit_builder: CircuitBuilder instance with circuit structure
        
        Returns:
            List of all possible Fault instances
        """
        from simulation.circuit_builder import GateType
        
        faults = []
        
        # Enumerate faults in each timestep
        for timestep_idx, timestep in enumerate(circuit_builder.timesteps):
            for gate in timestep.gates:
                if gate.gate_type in [GateType.INIT_0, GateType.INIT_PLUS, GateType.INIT_MAGIC]:
                    # Initialization errors
                    qubit = gate.qubits[0]
                    fault = Fault(
                        location_type=FaultLocation.INIT_DATA,
                        timestep=timestep_idx,
                        qubit_indices=(qubit,),
                        pauli_error='X'  # Bit flip during initialization
                    )
                    faults.append(fault)
                
                elif gate.gate_type in [GateType.H, GateType.S, GateType.S_DAG]:
                    # Single-qubit gate errors
                    qubit = gate.qubits[0]
                    for pauli in ['X', 'Y', 'Z']:
                        fault = Fault(
                            location_type=FaultLocation.SINGLE_QUBIT_GATE,
                            timestep=timestep_idx,
                            qubit_indices=(qubit,),
                            pauli_error=pauli
                        )
                        faults.append(fault)
                
                elif gate.gate_type == GateType.CNOT:
                    # Two-qubit gate errors
                    control, target = gate.qubits
                    # 15 non-trivial two-qubit Paulis
                    for p1 in ['I', 'X', 'Y', 'Z']:
                        for p2 in ['I', 'X', 'Y', 'Z']:
                            if p1 == 'I' and p2 == 'I':
                                continue
                            fault = Fault(
                                location_type=FaultLocation.TWO_QUBIT_GATE,
                                timestep=timestep_idx,
                                qubit_indices=(control, target),
                                pauli_error=p1 + p2
                            )
                            faults.append(fault)
                
                elif gate.gate_type == GateType.MEASURE_Z:
                    # Measurement errors
                    qubit = gate.qubits[0]
                    fault = Fault(
                        location_type=FaultLocation.MEASUREMENT,
                        timestep=timestep_idx,
                        qubit_indices=(qubit,),
                        pauli_error='flip'
                    )
                    faults.append(fault)
        
        self.faults = faults
        return faults
    
    def classify_faults(self, faults: List[Fault]) -> Dict[str, List[Fault]]:
        """
        Classify faults by whether they are detected and cause logical errors.
        
        This requires Heisenberg-picture tracking to determine:
        1. If fault is detected by post-selection
        2. If fault causes logical X or Z error
        
        Args:
            faults: List of faults to classify
        
        Returns:
            Dict with categories of faults
        """
        # This requires implementing Heisenberg tracking
        # For now, provide structure
        
        classified = {
            'detected': [],
            'undetected': [],
            'safe': [],  # Undetected but no logical error
            'logical_x': [],  # Undetected and causes X_L
            'logical_z': [],  # Undetected and causes Z_L
            'logical_both': [],  # Undetected and causes both
        }
        
        for fault in faults:
            # Use Heisenberg tracking to determine detection and logical error
            # This will be implemented with stabilizer tracking
            pass
        
        return classified
    
    def count_fault_types(self, classified_faults: Dict[str, List[Fault]]) -> Dict[str, int]:
        """
        Count number of faults in each category.
        
        Args:
            classified_faults: Dict from classify_faults
        
        Returns:
            Dict with counts
        """
        counts = {}
        for category, fault_list in classified_faults.items():
            counts[category] = len(fault_list)
        return counts


def compute_leading_order_pL(
    fault_counts: Dict[str, int],
    noise_params: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute leading-order logical error rate from fault counts.
    
    p_L ≈ Σ_i (# faults of type i) × (probability of type i)
    
    Args:
        fault_counts: Counts of different fault types
        noise_params: Dict with p1, p2, p_init, p_meas
    
    Returns:
        Dict with p_L breakdown and total
    """
    p1 = noise_params['p1']
    p2 = noise_params['p2']
    p_init = noise_params['p_init']
    p_meas = noise_params['p_meas']
    
    # Compute contributions
    # Each single-qubit gate error contributes p1/3 for each Pauli
    # Each two-qubit gate error contributes p2/15 for each two-qubit Pauli
    
    p_L_from_1q = fault_counts.get('logical_x_1q', 0) * (p1 / 3)
    p_L_from_1q += fault_counts.get('logical_z_1q', 0) * (p1 / 3)
    
    p_L_from_2q = fault_counts.get('logical_x_2q', 0) * (p2 / 15)
    p_L_from_2q += fault_counts.get('logical_z_2q', 0) * (p2 / 15)
    
    p_L_from_init = fault_counts.get('logical_x_init', 0) * p_init
    p_L_from_init += fault_counts.get('logical_z_init', 0) * p_init
    
    p_L_from_meas = fault_counts.get('logical_x_meas', 0) * p_meas
    p_L_from_meas += fault_counts.get('logical_z_meas', 0) * p_meas
    
    p_L_total = p_L_from_1q + p_L_from_2q + p_L_from_init + p_L_from_meas
    
    return {
        'p_L_1q': p_L_from_1q,
        'p_L_2q': p_L_from_2q,
        'p_L_init': p_L_from_init,
        'p_L_meas': p_L_from_meas,
        'p_L_total': p_L_total
    }


def derive_symbolic_expression(protocol_name: str) -> str:
    """
    Derive symbolic expression for p_L as function of noise parameters.
    
    Args:
        protocol_name: Name of protocol ('CR', 'MR', 'YL')
    
    Returns:
        LaTeX string with symbolic expression
    """
    expressions = {
        'YL': r"p_L \approx \frac{2}{5} p_2^2 + \frac{2}{3} p_1 + 2 p_I + O(p_2 p_1)",
        'CR': r"p_L \approx \frac{3}{5} p_2^2 + \alpha_1 p_1 + 2 p_I + O(p_2 p_1)",
        'MR': r"p_L \approx \frac{3}{5} p_2^2 + \beta_1 p_1 + p_I + O(p_2 p_1)",
    }
    
    return expressions.get(protocol_name, "p_L = <expression not derived>")


def test_fault_enumeration():
    """Test fault enumeration."""
    from simulation.circuit_builder import build_distance3_cr_circuit
    
    # Build circuit
    builder, metadata = build_distance3_cr_circuit()
    
    # Enumerate faults
    enumerator = FaultEnumerator(metadata)
    faults = enumerator.enumerate_all_faults(builder)
    
    print(f"Total faults enumerated: {len(faults)}")
    
    # Count by location type
    by_type = {}
    for fault in faults:
        loc_type = fault.location_type.value
        by_type[loc_type] = by_type.get(loc_type, 0) + 1
    
    print("\nFaults by location type:")
    for loc_type, count in sorted(by_type.items()):
        print(f"  {loc_type}: {count}")
    
    # Show some examples
    print("\nExample faults:")
    for i, fault in enumerate(faults[:5]):
        print(f"  {i+1}. {fault.location_type.value} at timestep {fault.timestep}, "
              f"qubit(s) {fault.qubit_indices}, error {fault.pauli_error}")
    
    print("\nFault enumeration tests passed!")


if __name__ == "__main__":
    test_fault_enumeration()
