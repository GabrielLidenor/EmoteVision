import pennylane as qp
import torch.nn as nn
from qiskit_ibm_runtime import QiskitRuntimeService
import os

class QuantumCircuit(nn.Module):
    def __init__(self, num_of_qubits, num_layers = 3, run_on_ibm = False):
        super().__init__()
        self.num_of_qubits = num_of_qubits
        self.device = self.configure_ibm_backend() if run_on_ibm else self.configure_local_simulation()
        diff_method = "parameter-shift" if run_on_ibm else "backprop"

        @qp.qnode(self.device, interface='torch', diff_method=diff_method)
        def _circuit(inputs, weights):
            qp.AngleEmbedding(inputs, wires=range(num_of_qubits), rotation='Y')
            qp.StronglyEntanglingLayers(weights, wires=range(num_of_qubits))

            return [qp.expval(qp.PauliZ(i)) for i in range(num_of_qubits)]

        weight_shapes = {"weights": (num_layers, num_of_qubits, 3)}
        self.qlayer = qp.qnn.TorchLayer(_circuit, weight_shapes)

    def forward(self, inputs):
        return self.qlayer(inputs)

    def configure_local_simulation(self):
         return qp.device(
            'default.qubit',
            wires = self.num_of_qubits,
        )

    def configure_ibm_backend(self):
        if not os.getenv("IBM_QUANTUM_TOKEN"):
            raise ValueError("No IBM Token found. Please add your IBM Token")

        service = QiskitRuntimeService(channel = "ibm_quantum_platform", token = os.getenv("IBM_QUANTUM_TOKEN"))

        print("Checking available backend at IBM Quantum")
        ibm_backend = service.least_busy(simulator = False, operational = True, min_num_qubits = self.num_of_qubits)
        print(f"Connecting to {ibm_backend.name}")

        return qp.device(
            'qiskit.remote',
            wires = self.num_of_qubits,
            backend = ibm_backend
        )
