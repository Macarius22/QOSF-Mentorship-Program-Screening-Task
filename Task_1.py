import numpy as np
import time
import matplotlib.pyplot as plt

# Define quantum gates
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)

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



# Run a small circuit simulation with multiple gates
def run_circuit(num_qubits):
    state = initialize_state(num_qubits)
    # Apply H gate to qubit 0
    state = apply_gate(H, state, 0, num_qubits)
    # Apply X gate to qubit 1
    state = apply_gate(X, state, 1, num_qubits)

    return state

# Measure runtime of the circuit with sequential gates
num_qubits_list = range(1, 13)
times = []

for num_qubits in num_qubits_list:
    start = time.time()
    state = run_circuit(num_qubits)
    end = time.time()
    times.append(end - start)




# Initialize the state tensor
def initialize_tensor_state(n):
    state_tensor = np.zeros([2] * n, dtype=complex)
    state_tensor[(0,) * n] = 1
    return state_tensor

# Apply a single-qubit gate to a tensor state
def apply_tensor_gate(gate, state_tensor, target_qubit):
    axes = ([1], [target_qubit])  # Contract gate with target qubit
    return np.tensordot(gate, state_tensor, axes=axes).reshape(state_tensor.shape)


# Measure the runtime of the tensor-based circuit simulation
times_tensor = []
for num_qubits in num_qubits_list:
    state_tensor = initialize_tensor_state(num_qubits)
    start = time.time()
    # Example: Apply X gate to qubit 0
    state_tensor = apply_tensor_gate(X, state_tensor, 0)
    end = time.time()
    times_tensor.append(end - start)

# Plot runtime comparison between naive and tensor-based simulations
plt.plot(num_qubits_list, times, label="Naive Simulation")
plt.plot(num_qubits_list, times_tensor, label="Tensor Simulation")
plt.xlabel("Number of Qubits")
plt.ylabel("Runtime (s)")
plt.title("Runtime of Naive vs Tensor Simulation")
plt.legend()
plt.show()
