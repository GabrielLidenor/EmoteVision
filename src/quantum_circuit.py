import pennylane as qp
import torch.nn as nn
from qiskit_ibm_runtime import QiskitRuntimeService
class QuantumCircuit(nn.Module):
    def __init__(self, num_of_qubits, num_layers = 3):
        super().__init__()
        self.num_of_qubits = num_of_qubits
        service = QiskitRuntimeService(channel = "ibm_quantum_platform", token="")

        print("Checking available backend at IBM Quantum")
        ibm_backend = service.least_busy(simulator = False, operational = True, min_num_qubits = num_of_qubits)
        print(f"Connecting to {ibm_backend.name}")

        self.device = qp.device(
            'qiskit.remote',
            wires = num_of_qubits,
            backend = ibm_backend
        )

        @qp.qnode(self.device, interface='torch', diff_method='parameter-shift')
        def _circuit(inputs, weights):
            qp.AngleEmbedding(inputs, wires=range(num_of_qubits), rotation='Y')
            qp.StronglyEntanglingLayers(weights, wires=range(num_of_qubits))

            return [qp.expval(qp.PauliZ(i)) for i in range(num_of_qubits)]

        weight_shapes = {"weights": (num_layers, num_of_qubits, 3)}
        self.qlayer = qp.qnn.TorchLayer(_circuit, weight_shapes)

    def forward(self, inputs):
        return self.qlayer(inputs)