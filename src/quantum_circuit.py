import pennylane as qp

class QuantumCircuit:
    def __init__(self, num_of_qubits):
        self.num_of_qubits = num_of_qubits
        self.device = qp.device('default.qubit', wires=num_of_qubits)

        self.quantum_function = qp.QNode(
            self._circuit,
            self.device,
            interface='torch',
            diff_method='backprop'
        )

    def _circuit(self, inputs, weights):
        qp.AngleEmbedding(inputs, wires=range(self.num_of_qubits), rotation='Y')
        qp.StronglyEntanglingLayers(weights, wires=range(self.num_of_qubits))

        return [qp.expval(qp.PauliZ(i)) for i in range(self.num_of_qubits)]
