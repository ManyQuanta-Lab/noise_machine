"""A small GHZ benchmark: sequential gates versus cluster feed-forward.

Run ``python ghz_mvp.py`` from the repository root.  Parameters live near the
bottom of this file so the first experiment stays easy to edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.classical import expr
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error


@dataclass(frozen=True)
class NoiseSettings:
    """Probabilities for the deliberately small MVP noise model."""

    cz_phase_error: float = 0.003
    idle_phase_error: float = 0.0005


def ideal_ghz(n_qubits: int) -> Statevector:
    """Return (|00...0> + |11...1>) / sqrt(2)."""
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for qubit in range(1, n_qubits):
        circuit.cx(0, qubit)
    return Statevector(circuit)


def sequential_ghz(n_qubits: int) -> QuantumCircuit:
    """Prepare GHZ by adding one output atom at a time."""
    circuit = QuantumCircuit(n_qubits, name="sequential")
    circuit.h(0)
    for qubit in range(1, n_qubits):
        append_cx_via_cz(circuit, qubit - 1, qubit)
    return circuit


def append_cx_via_cz(circuit: QuantumCircuit, control: int, target: int) -> None:
    """Append a CNOT using the MVP's native H-CZ-H gate sequence."""
    circuit.h(target)
    circuit.cz(control, target)
    circuit.h(target)


def cluster_feedforward_ghz(n_qubits: int) -> QuantumCircuit:
    """Prepare GHZ from a linear cluster, X measurements, and corrections.

    Measured odd-indexed cluster atoms are reset, then reused as output atoms.
    The correction is represented directly as Qiskit dynamic control flow.
    """
    if n_qubits < 2:
        raise ValueError("GHZ requires at least two qubits")

    cluster_length = n_qubits if n_qubits % 2 else n_qubits - 1
    ancillas = list(range(1, cluster_length - 1, 2))
    qubits = QuantumRegister(n_qubits, "q")
    measurements = ClassicalRegister(len(ancillas), "m")
    circuit = QuantumCircuit(qubits, measurements, name="cluster_feedforward")

    for qubit in range(cluster_length):
        circuit.h(qubits[qubit])
    for first in (0, 1):
        for qubit in range(first, cluster_length - 1, 2):
            circuit.cz(qubits[qubit], qubits[qubit + 1])

    for bit, ancilla in enumerate(ancillas):
        circuit.h(qubits[ancilla])
        circuit.measure(qubits[ancilla], measurements[bit])

    for output_index in range(1, len(ancillas) + 1):
        parity = expr.lift(measurements[0])
        for bit in range(1, output_index):
            parity = expr.bit_xor(parity, measurements[bit])
        with circuit.if_test(parity):
            circuit.x(qubits[2 * output_index])

    for ancilla in ancillas:
        circuit.reset(qubits[ancilla])
    for qubit in range(0, n_qubits - 1, 2):
        append_cx_via_cz(circuit, qubits[qubit], qubits[qubit + 1])
    return circuit


def idle_rounds(protocol: str, n_qubits: int) -> int:
    """Small timing proxy pending a real scheduler.

    The sequential chain has one additional waiting round for each added atom.
    The cluster protocol has a fixed two-round resource-state preparation.
    """
    if protocol == "sequential":
        return max(0, n_qubits - 2)
    if protocol == "cluster":
        return 2
    raise ValueError(f"unknown protocol: {protocol}")


def add_idle_proxy(circuit: QuantumCircuit, rounds: int) -> QuantumCircuit:
    """Append identity operations so the shared noise model can charge idling."""
    noisy_circuit = circuit.copy()
    for _ in range(rounds):
        for qubit in range(noisy_circuit.num_qubits):
            noisy_circuit.id(qubit)
    return noisy_circuit


def noise_model(settings: NoiseSettings) -> NoiseModel:
    """Apply the same channels to operations in both protocol circuits."""
    model = NoiseModel()
    cz_error = pauli_error(
        [("II", 1 - settings.cz_phase_error),
         ("IZ", settings.cz_phase_error / 3),
         ("ZI", settings.cz_phase_error / 3),
         ("ZZ", settings.cz_phase_error / 3)]
    )
    idle_error = pauli_error(
        [("I", 1 - settings.idle_phase_error), ("Z", settings.idle_phase_error)]
    )
    model.add_all_qubit_quantum_error(cz_error, "cz")
    model.add_all_qubit_quantum_error(idle_error, "id")
    return model


def fidelity(circuit: QuantumCircuit, n_qubits: int, settings: NoiseSettings | None = None) -> float:
    """Simulate the final density matrix and return its GHZ fidelity."""
    simulator = AerSimulator(method="density_matrix", noise_model=noise_model(settings) if settings else None)
    saved = circuit.copy()
    saved.save_density_matrix()
    result = simulator.run(transpile(saved, simulator)).result()
    final_state = DensityMatrix(result.data(0)["density_matrix"])
    return float(state_fidelity(final_state, ideal_ghz(n_qubits)))


def run_sweep(sizes: range, settings: NoiseSettings) -> pd.DataFrame:
    """Return one row per protocol and GHZ size."""
    rows: list[dict[str, object]] = []
    builders = {"sequential": sequential_ghz, "cluster": cluster_feedforward_ghz}
    for n_qubits in sizes:
        for name, build in builders.items():
            circuit = add_idle_proxy(build(n_qubits), idle_rounds(name, n_qubits))
            rows.append(
                {
                    "protocol": name,
                    "n_qubits": n_qubits,
                    "fidelity": fidelity(circuit, n_qubits, settings),
                    "cz_gates": circuit.count_ops().get("cz", 0),
                    "idle_rounds": idle_rounds(name, n_qubits),
                }
            )
    return pd.DataFrame(rows)


def plot_results(results: pd.DataFrame, output: Path) -> None:
    """Save the MVP comparison plot."""
    figure, axis = plt.subplots(figsize=(6, 4))
    for name, group in results.groupby("protocol"):
        axis.plot(group["n_qubits"], group["fidelity"], marker="o", label=name)
    axis.set(xlabel="GHZ qubits", ylabel="GHZ fidelity", ylim=(0, 1))
    axis.legend(title="protocol")
    figure.tight_layout()
    figure.savefig(output, dpi=200)


if __name__ == "__main__":
    # Density-matrix simulation grows exponentially; 2..6 is a responsive
    # default while we are still refining the model.
    SIZES = range(2, 7)
    SETTINGS = NoiseSettings()

    # Ticket MVP-2: both constructions must produce the ideal state without noise.
    for size in SIZES:
        for builder in (sequential_ghz, cluster_feedforward_ghz):
            score = fidelity(builder(size), size)
            if not np.isclose(score, 1.0):
                raise RuntimeError(f"{builder.__name__} failed ideal check for n={size}: {score}")

    results = run_sweep(SIZES, SETTINGS)
    print(results.to_string(index=False))
    plot_results(results, Path("ghz_mvp_fidelity.png"))
    print("Saved ghz_mvp_fidelity.png")
