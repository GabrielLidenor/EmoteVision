import torch
import pennylane as qp
import pytest

from src import QuantumCircuit
from src import quantum_circuit as quantum_circuit_module


def test_local_simulation_runs_and_returns_one_measurement_per_qubit():
    num_of_qubits = 2
    circuit = QuantumCircuit(num_of_qubits=num_of_qubits, num_layers=1)

    output = circuit(torch.zeros((1, num_of_qubits)))

    assert circuit.device.name == "default.qubit"
    assert output.shape == (1, num_of_qubits)


def test_local_simulation_backpropagates_for_batched_inputs():
    num_of_qubits = 2
    circuit = QuantumCircuit(num_of_qubits=num_of_qubits, num_layers=1)
    inputs = torch.zeros((2, num_of_qubits), requires_grad=True)

    circuit(inputs).sum().backward()

    assert inputs.grad is not None
    assert circuit.qlayer.weights.grad is not None


def test_ibm_backend_requires_a_token(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)

    with pytest.raises(ValueError, match="IBM Token"):
        QuantumCircuit(num_of_qubits=2, run_on_ibm=True)


def test_ibm_backend_is_selected_without_contacting_ibm(monkeypatch):
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "test-token")

    class FakeBackend:
        name = "fake-backend"

    backend = FakeBackend()
    service_options = {}
    backend_options = {}
    device_options = {}
    real_device = qp.device

    class FakeRuntimeService:
        def __init__(self, **kwargs):
            service_options.update(kwargs)

        def least_busy(self, **kwargs):
            backend_options.update(kwargs)
            return backend

    def fake_device(name, **kwargs):
        device_options.update(name=name, **kwargs)
        return real_device("default.qubit", wires=kwargs["wires"])

    monkeypatch.setattr(
        quantum_circuit_module,
        "QiskitRuntimeService",
        FakeRuntimeService,
    )
    monkeypatch.setattr(quantum_circuit_module.qp, "device", fake_device)

    circuit = QuantumCircuit(num_of_qubits=2, num_layers=1, run_on_ibm=True)

    assert service_options == {
        "channel": "ibm_quantum_platform",
        "token": "test-token",
    }
    assert backend_options == {
        "simulator": False,
        "operational": True,
        "min_num_qubits": 2,
    }
    assert device_options == {
        "name": "qiskit.remote",
        "wires": 2,
        "backend": backend,
    }
    assert circuit.device.name == "default.qubit"
