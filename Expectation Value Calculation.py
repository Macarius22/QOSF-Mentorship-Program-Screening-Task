import numpy as np

# Define quantum gates
I = np.array([[1, 0], [0, 1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Initialize an n-qubit |0...0> state
def initialize_state(n):
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1
    return state

# Apply a single-qubit gate to a specific qubit in the state vector
def apply_gate(gate, state, target_qubit, num_qubits):
    full_gate = 1
    for i in range(num_qubits):
        if i == target_qubit:
            full_gate = np.kron(full_gate, gate)
        else:
            full_gate = np.kron(full_gate, I)
    return full_gate @ state

# Define a simple 3-qubit circuit that applies H to each qubit
def run_circuit(num_qubits):
    state = initialize_state(num_qubits)
    # Apply H gate to each qubit to create an equal superposition
    for qubit in range(num_qubits):
        state = apply_gate(H, state, qubit, num_qubits)
    return state

# Compute the expectation value for an operator
def compute_expectation(state, operator):
    # Conjugate transpose of the state
    state_dagger = np.conjugate(state).T
    # Compute the expectation value
    expectation_value = state_dagger @ operator @ state
    return expectation_value.real  # Return only the real part

# Run the circuit
num_qubits = 3
final_state = run_circuit(num_qubits)

# Define an operator, e.g., Z operator on the first qubit of a 3-qubit system
operator = np.kron(Z, np.kron(I, I))  # Z on qubit 0, identity on qubits 1 and 2

# Compute expectation value
expectation_value = compute_expectation(final_state, operator)
print("Expectation value:", expectation_value)
