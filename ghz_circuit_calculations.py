import qiskit
from qiskit.visualization import circuit_drawer
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit.quantum_info import Statevector, SparsePauliOp, Operator
import numpy as np
from qiskit.circuit.classical import expr
from qiskit_aer.noise import NoiseModel, pauli_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from itertools import product
import matplotlib.ticker as ticker

###############################################################################
# Parameters
###############################################################################

num_qubits = np.arange(2, 11)
num_qubits_theory = np.arange(2, 22)
cz_error_prob = 0.003
idling_error_prob = 0.0005

# Note that the measurement errors and reset errors are zero.

###############################################################################
# Functions for generating the circuits
###############################################################################

def ghz(N) -> qiskit.QuantumCircuit:
    qr = qiskit.QuantumRegister(N)
    qc = qiskit.QuantumCircuit(qr)

    shift = Operator(np.eye(2**N))

    for i in range(N):
        qc.h(qr[i])
    
    qc.cz(qr[N//2-1], qr[N//2])

    for layer in range(0,N//2):
        a = N//2+layer
        b = a+1

        c = N//2-1-layer
        d = c-1

        if a < N and b < N:
            qc.h(qr[a])
        
        qc.unitary(shift, range(N), label='shift')

        if a < N and b < N:
            qc.cz(qr[a], qr[b])

        if c >= 0 and d >= 0:
            qc.cz(qr[c], qr[d])
            qc.h(qr[d])
    
    qc.h(qr[N-1])

    return qc

def ghz_from_cluster(N) -> qiskit.QuantumCircuit:
    qr = qiskit.QuantumRegister(N)
    cr = qiskit.ClassicalRegister((N-1)//2)
    qc = qiskit.QuantumCircuit(qr, cr)

    shift = Operator(np.eye(2**N))

    # Step 1: Create 1D cluster state of 2n qubits
    for i in range(0,N,2):
        qc.h(qr[i])
    for i in range(1,N-1,2):
        qc.h(qr[i])
    for i in range(0,N - 2,2):
        qc.cz(qr[i], qr[i + 1])
    for i in range(1,N - 1,2):
        if i == 1:
            qc.unitary(shift, range(N), label='shift')
        qc.cz(qr[i], qr[i + 1])
    
    qc.barrier(qr)

    # Step 2: Measure even qubits (2,4,...) in X basis
    for k in range(1, N-1, 2):
        qc.h(qr[k])
        qc.measure(qr[k], cr[k // 2])
    
    qc.barrier(qr)

    # Step 3: Conditional Z corrections on right neighbor (odd qubits)
    parity_expr = None
    for k in range(1, N-1, 2):
        if parity_expr is None:
            parity_expr = expr.lift(cr[k // 2])
        else:
            parity_expr = expr.bit_xor(parity_expr, cr[k // 2])

        with qc.if_test(parity_expr):
            qc.x(qr[k+1])
    
    qc.barrier(qr)
    
    # Step 4: Reset the measured qubits to zero
    for k in range(1, N - 1, 2):
        qc.reset(qr[k])
    
    qc.barrier(qr)

    # Step 5: Add the even qubits to the cluster
    for i in range(0,N - 1,2):
        qc.h(qr[i+1])
    for i in range(0,N - 1,2):
        if i == 0 and N > 2:
            qc.unitary(shift, range(N), label='shift')
        qc.cz(qr[i], qr[i + 1])
        qc.h(qr[i+1])

    return qc

###############################################################################
# Functions for simulating the circuit to obtain 
# a) the fidelity and
# b) the final state vector (this function is not used in the following)
###############################################################################

def simulate_fidelity(qc: qiskit.QuantumCircuit, p_cz, p_idl, shots=5000) -> Statevector:
    ghz_state = np.zeros(2**qc.num_qubits)
    ghz_state[[0,-1]] = 1/np.sqrt(2)
    projector = np.outer(ghz_state, ghz_state.conj())
    observable = SparsePauliOp.from_operator(projector)

    # Biased Z noise, equation 2 of https://arxiv.org/pdf/2505.15669
    noise_model = NoiseModel()

    error_cz = pauli_error([
        ('II', 1 - p_cz),
        ('IZ', p_cz/3),
        ('ZI', p_cz/3),
        ('ZZ', p_cz/3),
    ])
    noise_model.add_all_qubit_quantum_error(error_cz, 'cz')

    z1 = pauli_error([('Z', p_idl), ('I', 1-p_idl)])
    error_shift = z1
    for _ in range(qc.num_qubits-1):
        error_shift = error_shift.tensor(z1)
    noise_model.add_all_qubit_quantum_error(error_shift, 'shift')

    estimator = EstimatorV2(options={"backend_options": {"method":"statevector", "shots":shots, "noise_model":noise_model, "matrix_product_state_max_bond_dimension":4}}) # statevector, matrix_product_state
    result = estimator.run([(qc,observable)]).result()[0]
    return float(result.data.evs)

def simulate_statevector(qc: qiskit.QuantumCircuit) -> Statevector:
    qc_sv = qc.copy()
    qc_sv.save_statevector()

    sim = AerSimulator(method='automatic')
    tqc = qiskit.transpile(qc_sv, sim)
    result = sim.run(tqc, shots=1).result()

    psi = result.data(0)["statevector"]
    return psi

###############################################################################
# Functions for calculating the fidelity analytically
###############################################################################

def analytic_fidelity_ghz_standard(n: int, p_cz: float, p_idl: float) -> float:
    num_two_qubit_gates = n-1
    number_of_shifts = n // 2
    number_of_shifts[n < 3] = 0

    # After the Hadamards, the errors are IX, ZI, ZX.
    # An odd number of Z / any X takes us out of the stabilizer space.
    return ((1-p_cz/3)*(1-p_cz/3)*(1-p_cz/3))**num_two_qubit_gates  * (1-p_idl)**(n*number_of_shifts)

def analytic_fidelity_ghz_measurement_based(n: int, p_cz: float, p_idl: float) -> float:
    num_two_qubit_gates = n - 1 + (n - 1)//2
    number_of_shifts = 2*np.ones_like(n)
    number_of_shifts[n < 3] = 0
    print(n, num_two_qubit_gates)

    return ((1-p_cz/3)*(1-p_cz/3)*(1-p_cz/3))**num_two_qubit_gates * (1-p_idl)**(n*number_of_shifts)

###############################################################################
# Run the code
###############################################################################

# Circuits for creating a ghz state
qc = ghz_from_cluster(8)
file_path = "ghz_circuit_measurement_based.png"
circuit_drawer(qc, output='mpl', filename=file_path, fold=-1)
print(f"Circuit saved to {file_path}")

qc = ghz(8)
file_path = "ghz_circuit_standard.png"
circuit_drawer(qc, output='mpl', filename=file_path, fold=-1)
print(f"Circuit saved to {file_path}")

# Calculate the fidelity
combos = list(product(num_qubits, ["standard", "measurement based"], range(10)))
np.random.shuffle(combos)

fidelities = []
for n, method, _ in tqdm(combos):
    if method == "standard":
        qc = ghz(n)
    else:
        qc = ghz_from_cluster(n)
    fidelities.append({
        "n": n,
        "method": method,
        "fidelity": simulate_fidelity(qc, cz_error_prob, idling_error_prob),
    })
df = pd.DataFrame(fidelities)

# Plot the results
plt.figure(figsize=(6, 4))

sns.lineplot(
    data=df,
    x="n",
    y="fidelity",
    hue="method",
    hue_order=["standard", "measurement based"],
    estimator="mean",
    errorbar=("ci", 95),
)

plt.plot(
    num_qubits_theory,
    analytic_fidelity_ghz_standard(num_qubits_theory, cz_error_prob, idling_error_prob),
    "k--",
)

plt.plot(
    num_qubits_theory,
    analytic_fidelity_ghz_measurement_based(num_qubits_theory, cz_error_prob, idling_error_prob),
    "k--",
    label="analytic",
)
plt.title(f"$IZ$, $ZI$, $ZZ$ on CZ gates with probability $p_{{CZ}}/3={cz_error_prob/3:.3}$ each,\n $Z$ idling errors with $p_{{idl}}={idling_error_prob}$ per shift.")
plt.legend(title=None)
plt.xlabel("Number of qubits $n$")
plt.ylabel(r"$F=\overline{|\langle \Psi |GHZ \rangle |^2}$")
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.gca().margins(x=0)
plt.tight_layout()
plt.savefig("fidelity_vs_qubit_number.png", dpi=300)
print("Plot saved to fidelity_vs_n.png")
