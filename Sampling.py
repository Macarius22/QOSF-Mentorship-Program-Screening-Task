import numpy as np

# Define quantum gates
I = np.array([[1, 0], [0, 1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)

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

# Sampling function from the final state vector
def sample_state(state, num_samples=1000):
    # Calculate probabilities from state vector
    probabilities = np.abs(state) ** 2
    
    # Generate samples from the probability distribution
    outcomes = np.random.choice(len(state), size=num_samples, p=probabilities)
    
    # Count the occurrences of each basis state
    counts = {format(i, f'0{int(np.log2(len(state)))}b'): np.sum(outcomes == i) for i in range(len(state))}
    return counts

# Run the circuit
num_qubits = 3
final_state = run_circuit(num_qubits)

# Sample from the final state
sample_counts = sample_state(final_state, num_samples=1000)

# Output the sampled outcomes
print("Sampled outcomes:", sample_counts)
