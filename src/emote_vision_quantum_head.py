import torch
import torch.nn as nn
import pennylane as qp
from src import QuantumCircuit
import torch.nn.functional as F

class EmoteVisionQuantumHead(nn.Module):
    def __init__(self, in_features, number_of_layers, number_of_qubits, number_of_classes):
        super().__init__()
        self.reducer = nn.Linear(in_features, number_of_qubits)

        quantum_circuit = QuantumCircuit(number_of_qubits)
        self.qlayer = qp.qnn.TorchLayer(quantum_circuit.quantum_function, {'weights': (number_of_layers, number_of_qubits, 3)})
        self.fc_final = nn.Linear(number_of_qubits, number_of_classes)

    def forward(self, x):
        reduced_features = self.reducer(x)
        rotation_angles = torch.tanh(reduced_features) * torch.pi
        quantum_measurements = self.qlayer(rotation_angles)
        logits = self.fc_final(quantum_measurements)

        return logits