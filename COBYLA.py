from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit_aer.primitives import Estimator as AerEstimator

import matplotlib.pyplot as plt
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_algorithms.optimizers import COBYLA


# =========================
# 1. GRAFO
# =========================
edges = [(0, 1), (0, 2), (1, 2), (0, 4), (2, 3), (3, 4)]
num_nodes = 5

graph = rx.PyGraph()
graph.add_nodes_from(range(num_nodes))
graph.add_edges_from([(u, v, 1.0) for (u, v) in edges])

draw_graph(graph, node_size=600, with_labels=True)
plt.show()


# =========================
# 2. HAMILTONIANA MAX-CUT
# =========================
# H_C = 0.5 * sum_(i,j) (Z_i Z_j - I)
pauli_terms = []
coeffs = []

for (i, j) in edges:
    z_string = ["I"] * num_nodes
    z_string[i] = "Z"
    z_string[j] = "Z"
    pauli_terms.append("".join(z_string))
    coeffs.append(0.5)

pauli_terms.append("I" * num_nodes)
coeffs.append(-0.5 * len(edges))

cost_hamiltonian = SparsePauliOp(pauli_terms, coeffs)
print(cost_hamiltonian)


# =========================
# 3. PARAMETRI GLOBALI
# =========================
p = 3
shots_noisy = 1024
maxiter = 80
n_runs = 20

levels = [
    ("Ideal baseline", None),
    ("Readout low", [[0.99, 0.01], [0.02, 0.98]]),
    ("Readout medium", [[0.97, 0.03], [0.04, 0.96]]),
    ("Readout high", [[0.94, 0.06], [0.06, 0.94]]),
]


# =========================
# 4. ANSATZ
# =========================
ansatz = QAOAAnsatz(cost_hamiltonian, reps=p)
num_params = ansatz.num_parameters


# =========================
# 5. COST DA BITSTRING
# =========================
def maxcut_value(bitstring, edges):
    bits = bitstring[::-1]
    value = 0
    for i, j in edges:
        if bits[i] != bits[j]:
            value += 1
    return value


def expected_cost_from_counts(counts, edges, shots):
    exp_val = 0.0
    for bitstring, count in counts.items():
        exp_val += (-maxcut_value(bitstring, edges)) * (count / shots)
    return exp_val


# =========================
# 6. BACKEND RUMOROSI
# =========================
def build_noisy_simulator(readout_matrix=None):
    if readout_matrix is None:
        return AerSimulator()

    noise_model = NoiseModel()
    readout = ReadoutError(readout_matrix)

    for q in range(num_nodes):
        noise_model.add_readout_error(readout, [q])

    return AerSimulator(noise_model=noise_model)


# =========================
# 7. COST IDEALE ESATTA CON ESTIMATOR
# =========================
def ideal_cost_func(params, ansatz, hamiltonian, estimator, history):
    job = estimator.run(
        circuits=[ansatz],
        observables=[hamiltonian],
        parameter_values=[params]
    )
    value = job.result().values[0].real
    history.append(value)
    return value


# =========================
# 8. COST RUMOROSA DA COUNTS
# =========================
def noisy_cost_func(params, ansatz, simulator, edges, shots, history):
    qc = ansatz.assign_parameters(params)

    qc_meas = QuantumCircuit(num_nodes, num_nodes)
    qc_meas.compose(qc, inplace=True)
    qc_meas.measure(range(num_nodes), range(num_nodes))

    tqc = transpile(qc_meas, simulator)

    result = simulator.run(
        tqc,
        shots=shots
    ).result()

    counts = result.get_counts()
    value = expected_cost_from_counts(counts, edges, shots)

    history.append(value)
    return value


# =========================
# 9. DISTRIBUZIONE FINALE IDEALE
# =========================
def final_distribution_ideal(opt_params, ansatz):
    sim = AerSimulator()
    qc_opt = ansatz.assign_parameters(opt_params)

    qc_meas = QuantumCircuit(num_nodes, num_nodes)
    qc_meas.compose(qc_opt, inplace=True)
    qc_meas.measure(range(num_nodes), range(num_nodes))

    tqc = transpile(qc_meas, sim)
    result = sim.run(tqc, shots=shots_noisy).result()
    counts = result.get_counts()

    probs = {bitstring: count / shots_noisy for bitstring, count in counts.items()}
    return probs


# =========================
# 10. DISTRIBUZIONE FINALE RUMOROSA
# =========================
def final_distribution_noisy(opt_params, ansatz, simulator, shots):
    qc_opt = ansatz.assign_parameters(opt_params)

    qc_meas = QuantumCircuit(num_nodes, num_nodes)
    qc_meas.compose(qc_opt, inplace=True)
    qc_meas.measure(range(num_nodes), range(num_nodes))

    tqc = transpile(qc_meas, simulator)
    result = simulator.run(tqc, shots=shots).result()
    counts = result.get_counts()

    probs = {bitstring: count / shots for bitstring, count in counts.items()}
    return probs


# =========================
# 11. ESECUZIONE SINGOLA RUN
# =========================
def single_run(init_params_fixed):
    run_histories = {}
    run_final_values = {}
    run_final_params = {}
    run_final_distributions = {}
    run_best_bitstrings = {}

    for label, mat in levels:
        print(f"\n========== {label} ==========")

        history = []
        optimizer = COBYLA(maxiter=maxiter)

        if mat is None:
            print("Caso ideale con Estimator esatto")

            estimator = AerEstimator(
                approximation=True,
                run_options={"shots": None}
            )

            result = optimizer.minimize(
                fun=lambda params: ideal_cost_func(
                    params, ansatz, cost_hamiltonian, estimator, history
                ),
                x0=init_params_fixed.copy(),
            )

            opt_params = result.x
            opt_value = result.fun

            probs = final_distribution_ideal(opt_params, ansatz)

        else:
            print("Matrice di readout:")
            print(np.array(mat))

            simulator = build_noisy_simulator(mat)

            result = optimizer.minimize(
                fun=lambda params: noisy_cost_func(
                    params, ansatz, simulator, edges, shots_noisy, history
                ),
                x0=init_params_fixed.copy(),
            )

            opt_params = result.x
            opt_value = result.fun

            probs = final_distribution_noisy(opt_params, ansatz, simulator, shots_noisy)

        print("Parametri ottimali:", opt_params)
        print("Valore minimo medio <H_C>:", opt_value)

        best_bitstring = max(probs, key=probs.get)

        run_histories[label] = history
        run_final_values[label] = opt_value
        run_final_params[label] = opt_params
        run_final_distributions[label] = probs
        run_best_bitstrings[label] = best_bitstring

    return run_histories, run_final_values, run_final_params, run_final_distributions, run_best_bitstrings


# =========================
# 12. MULTI-RUN
# =========================
all_bitstrings = [format(i, f"0{num_nodes}b") for i in range(2**num_nodes)]

all_run_values = {label: [] for label, _ in levels}
all_run_histories = {label: [] for label, _ in levels}
all_run_distributions = {label: [] for label, _ in levels}
all_run_best_bitstrings = {label: [] for label, _ in levels}

for run_idx in range(n_runs):
    print(f"\n\n#############################")
    print(f"RUN {run_idx + 1}/{n_runs}")
    print(f"#############################")

    init_params_fixed = np.random.uniform(0, 1, size=num_params)

    run_histories, run_final_values, run_final_params, run_final_distributions, run_best_bitstrings = single_run(init_params_fixed)

    for label in run_final_values:
        all_run_values[label].append(run_final_values[label])
        all_run_histories[label].append(run_histories[label])
        all_run_distributions[label].append(run_final_distributions[label])
        all_run_best_bitstrings[label].append(run_best_bitstrings[label])


# =========================
# 13. STATISTICHE FINALI
# =========================
print("\n\n========== STATISTICHE FINALI ==========")

stats_summary = {}

for label in all_run_values:
    values = np.array(all_run_values[label])

    mean_val = np.mean(values)
    std_val = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    median_val = np.median(values)

    stats_summary[label] = {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "median": median_val,
    }

    print(f"\n{label}")
    print(f"  mean   = {mean_val}")
    print(f"  std    = {std_val}")
    print(f"  min    = {min_val}")
    print(f"  max    = {max_val}")
    print(f"  median = {median_val}")


# =========================
# 14. BARPLOT MEDIA ± STD
# =========================
labels = list(stats_summary.keys())
means = [stats_summary[label]["mean"] for label in labels]
stds = [stats_summary[label]["std"] for label in labels]

plt.figure(figsize=(9, 5))
plt.bar(labels, means, yerr=stds, capsize=6, color="tab:purple", alpha=0.8)
plt.xlabel("Caso")
plt.ylabel("Valore finale medio <H_C>")
plt.title("Media e deviazione standard del valore finale")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# =========================
# 15. BOXPLOT
# =========================
plt.figure(figsize=(10, 6))
data_for_boxplot = [all_run_values[label] for label in labels]
bp = plt.boxplot(data_for_boxplot, tick_labels=labels, patch_artist=True)

colors = ["lightblue", "lightgreen", "salmon", "plum"]
for patch, color in zip(bp["boxes"], colors[:len(labels)]):
    patch.set_facecolor(color)

plt.ylabel("Valore finale <H_C>")
plt.title("Distribuzione dei valori finali su più run")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# =========================
# 16. CURVA MEDIA DI CONVERGENZA
# =========================
plt.figure(figsize=(10, 6))

for label in labels:
    histories = all_run_histories[label]
    min_len = min(len(h) for h in histories)
    trimmed = np.array([h[:min_len] for h in histories])

    mean_history = np.mean(trimmed, axis=0)
    std_history = np.std(trimmed, axis=0)

    x = np.arange(min_len)
    plt.plot(x, mean_history, label=label)
    plt.fill_between(x, mean_history - std_history, mean_history + std_history, alpha=0.2)

plt.xlabel("Valutazione della cost function")
plt.ylabel("<H_C>")
plt.title("Curva media di convergenza con banda di deviazione standard")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# 17. DISTRIBUZIONE MEDIA DEI BITSTRING
# =========================
for label in labels:
    avg_probs = []

    for bitstring in all_bitstrings:
        values = [
            dist.get(bitstring, 0.0)
            for dist in all_run_distributions[label]
        ]
        avg_probs.append(np.mean(values))

    plt.figure(figsize=(14, 5))
    plt.bar(all_bitstrings, avg_probs, color="tab:green")
    plt.xlabel("Bitstring")
    plt.ylabel("Probabilità media finale")
    plt.title(f"Distribuzione media finale dei bitstring - {label}")
    plt.xticks(rotation=90)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================
# 18. BITSTRING PIÙ PROBABILE PER RUN
# =========================
for label in labels:
    best_list = all_run_best_bitstrings[label]
    unique_bs = sorted(set(best_list))
    frequencies = [best_list.count(bs) for bs in unique_bs]

    print(f"\n{label} - Most likely bitstring per run:")
    for bs, freq in zip(unique_bs, frequencies):
        print(f"  {bs}: {freq} volte")

    plt.figure(figsize=(10, 4))
    plt.bar(unique_bs, frequencies, color="tab:orange")
    plt.xlabel("Bitstring")
    plt.ylabel("Numero di run in cui è il più probabile")
    plt.title(f"Bitstring più probabile nelle varie run - {label}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()