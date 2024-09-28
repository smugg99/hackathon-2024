from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator 
from qiskit.visualization import plot_histogram
import time

# Representing "Hello, World!" in binary (ASCII values)
binary_hello_world = ''.join(format(ord(c), '08b') for c in "Hello, World!")
n_qubits = len(binary_hello_world)

# Create a quantum circuit with the number of qubits equal to the binary representation length
qc = QuantumCircuit(n_qubits, n_qubits)

# Apply X-gates to simulate the binary representation of "Hello, World!"
for i, bit in enumerate(binary_hello_world):
    if bit == '1':
        qc.x(i)  # Apply an X-gate if the bit is 1 (this flips the qubit from |0⟩ to |1⟩)

# Measure the qubits
qc.measure(range(n_qubits), range(n_qubits))

# Draw the circuit
qc.draw('mpl')

# Simulate the execution of the circuit
start_time = time.time()
simulator = AerSimulator()
sim_result = simulator.run(qc).result()
end_time = time.time()

# Extract the result
counts = sim_result.get_counts()
measured_binary = list(counts.keys())[0]

# Convert the measured binary back to "Hello, World!"
hello_world_output = ''.join(chr(int(measured_binary[i:i+8], 2)) for i in range(0, len(measured_binary), 8))

quantum_execution_time = end_time - start_time
print(f"Quantum 'Hello, World!': {hello_world_output}")
print(f"Quantum execution time: {quantum_execution_time} seconds")
