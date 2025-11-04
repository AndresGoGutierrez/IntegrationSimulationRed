"""
Sistema de Simulación de Red con Teoría de Colas, Cadenas de Markov y Bootstrap
=================================================================================

Este script orquesta la simulación completa del sistema de red,
integrando teoría de colas M/M/n, cadenas de Markov y validación estadística
mediante Bootstrap.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

from colas_simulacion import QueueSimulator
from cadenas_markov import MarkovChainSimulator
from analisis_boostrap import BootstrapAnalyzer
from visualizacion import Visualizer

# -----------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 10
})


# -----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Ejecuta la simulación completa del sistema de red.

    Pasos:
    1. Simulación de teoría de colas (M/M/n)
    2. Simulación de cadena de Markov
    3. Validación estadística con Bootstrap
    4. Visualizaciones
    5. Generación de informe final
    """
    print("=" * 80)
    print("SISTEMA DE SIMULACIÓN DE RED - ANÁLISIS COMPLETO")
    print("=" * 80, "\n")

    # -------------------------------------------------------------------------
    # PASO 1: CONFIGURACIÓN DE PARÁMETROS
    # -------------------------------------------------------------------------
    print("PASO 1: Configuración de parámetros del sistema")
    print("-" * 80)

    lambda_rate = 3.0   # Tasa de llegada (solicitudes/min)
    mu_rate = 4.0       # Tasa de servicio (solicitudes/min)
    n_servers = 3       # Número de servidores
    capacity = 10       # Capacidad máxima
    simulation_time = 1000  # Tiempo total (min)

    print(f"  • λ (tasa de llegada): {lambda_rate} solicitudes/min")
    print(f"  • μ (tasa de servicio): {mu_rate} solicitudes/min")
    print(f"  • Servidores (n): {n_servers}")
    print(f"  • Capacidad (K): {capacity}")
    print(f"  • Tiempo de simulación: {simulation_time} min\n")

    # -------------------------------------------------------------------------
    # PASO 2: SIMULACIÓN DE TEORÍA DE COLAS
    # -------------------------------------------------------------------------
    print("PASO 2: Simulación del sistema de colas M/M/n")
    print("-" * 80)

    queue_sim = QueueSimulator(lambda_rate, mu_rate, n_servers, capacity)
    queue_results = queue_sim.simulate(simulation_time)
    theoretical_metrics = queue_sim.calculate_theoretical_metrics()

    print("\nMétricas del Sistema de Colas:")
    print(f"  • ρ (utilización): {queue_results['rho']:.4f}")
    print(f"  • Wq (tiempo en cola): {queue_results['Wq']:.4f} min")
    print(f"  • Ws (tiempo total): {queue_results['Ws']:.4f} min")
    print(f"  • Lq (longitud cola): {queue_results['Lq']:.4f}")
    print(f"  • Ls (en sistema): {queue_results['Ls']:.4f}")
    print(f"  • Atendidas: {queue_results['served']} | Rechazadas: {queue_results['rejected']}")

    if theoretical_metrics:
        print("\nComparación Teórica:")
        print(f"  • ρ teórico: {theoretical_metrics['rho']:.4f}")
        print(f"  • Lq teórico: {theoretical_metrics['Lq']:.4f}")
        print(f"  • Wq teórico: {theoretical_metrics['Wq']:.4f}")

    print(f"\n{'Inestable (ρ ≥ 1)' if queue_results['rho'] >= 1 else 'Estable (ρ < 1)'}\n")

    # -------------------------------------------------------------------------
    # PASO 3: CADENA DE MARKOV
    # -------------------------------------------------------------------------
    print("PASO 3: Simulación de Cadena de Markov")
    print("-" * 80)

    transition_matrix = np.array([
        [0.85, 0.12, 0.03],  # Operativo
        [0.40, 0.50, 0.10],  # Degradado
        [0.30, 0.20, 0.50]   # Fallido
    ])
    state_names = ["Operativo", "Degradado", "Fallido"]

    markov_sim = MarkovChainSimulator(transition_matrix, state_names)
    n_steps = 500
    markov_results = markov_sim.simulate(n_steps=n_steps, initial_state=0)
    stationary_dist = markov_sim.calculate_stationary_distribution()

    print("\nDistribución Estacionaria:")
    for state, prob in zip(state_names, stationary_dist):
        print(f"  • {state}: {prob:.4f} ({prob * 100:.2f}%)")

    print("\nFrecuencias Observadas:")
    for state, count in markov_results["state_counts"].items():
        freq = count / n_steps
        print(f"  • {state}: {freq:.4f} ({freq * 100:.2f}%)")
    print()

    # -------------------------------------------------------------------------
    # PASO 4: ANÁLISIS DE BOOTSTRAP
    # -------------------------------------------------------------------------
    print("PASO 4: Validación estadística con Bootstrap")
    print("-" * 80)

    bootstrap_analyzer = BootstrapAnalyzer(n_bootstrap=1000, confidence_level=0.95)

    wait_times = queue_results["wait_times"]
    system_times = queue_results["system_times"]
    queue_lengths = queue_results["queue_length_history"]

    print(f"\n  Ejecutando {bootstrap_analyzer.n_bootstrap} iteraciones de bootstrap...")

    metrics_to_analyze = {
        "Wq": (wait_times, np.mean, "Tiempo promedio en cola (min)"),
        "Ws": (system_times, np.mean, "Tiempo total en sistema (min)"),
        "ρ": (queue_lengths, lambda x: np.mean(x) / n_servers, "Utilización del sistema")
    }

    bootstrap_results = {}
    for key, (data, func, label) in metrics_to_analyze.items():
        bootstrap_results[key] = bootstrap_analyzer.bootstrap_metric(data, func)
        stats = bootstrap_results[key]
        print(f"\n{label}:")
        print(f"  Media: {stats['mean']:.4f}")
        print(f"  IC95%: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
        print(f"  σ: {stats['std']:.4f}")

    # Coeficiente de variación
    for metric, results in bootstrap_results.items():
        cv = results["std"] / results["mean"]
        label = "Robusto" if cv < 0.1 else "Moderado" if cv < 0.2 else "Variable"
        print(f"  → CV({metric}): {cv:.4f} {label}")
    print()

    # -------------------------------------------------------------------------
    # PASO 5: VISUALIZACIONES
    # -------------------------------------------------------------------------
    print("PASO 5: Generación de visualizaciones")
    print("-" * 80)

    visualizer = Visualizer()
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.ravel()

    visualizer.plot_wait_time_distribution(wait_times, ax=axes[0])
    visualizer.plot_system_time_distribution(system_times, ax=axes[1])
    visualizer.plot_queue_length_evolution(queue_results["queue_length_history"], ax=axes[2])
    visualizer.plot_markov_evolution(markov_results["state_sequence"], state_names, ax=axes[3])

    observed_freq = [markov_results["state_counts"][s] / n_steps for s in state_names]
    visualizer.plot_stationary_distribution(stationary_dist, observed_freq, state_names, ax=axes[4])
    visualizer.plot_transition_matrix(transition_matrix, state_names, ax=axes[5])

    visualizer.plot_bootstrap_distribution(
        bootstrap_results["Wq"]["bootstrap_samples"],
        bootstrap_results["Wq"]["ci_lower"],
        bootstrap_results["Wq"]["ci_upper"],
        "Wq (min)",
        ax=axes[6]
    )
    visualizer.plot_bootstrap_distribution(
        bootstrap_results["Ws"]["bootstrap_samples"],
        bootstrap_results["Ws"]["ci_lower"],
        bootstrap_results["Ws"]["ci_upper"],
        "Ws (min)",
        ax=axes[7]
    )

    if theoretical_metrics:
        visualizer.plot_theoretical_vs_simulated(theoretical_metrics, queue_results, ax=axes[8])

    plt.tight_layout()
    plt.savefig("simulation_results.png", dpi=300, bbox_inches="tight")
    print("Visualizaciones guardadas en 'simulation_results.png'\n")

    # -------------------------------------------------------------------------
    # PASO 6: INFORME FINAL
    # -------------------------------------------------------------------------
    print("PASO 6: Generación de informe final")
    print("-" * 80)

    report_text = generate_report(
        queue_results,
        theoretical_metrics,
        markov_results,
        stationary_dist,
        bootstrap_results,
        parameters={
            "lambda": lambda_rate,
            "mu": mu_rate,
            "n_servers": n_servers,
            "capacity": capacity,
            "simulation_time": simulation_time,
        },
    )

    with open("simulation_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("Informe guardado en 'simulation_report.txt'\n")
    print("=" * 80)
    print("SIMULACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 80)
    print("\nArchivos generados:")
    print("  • simulation_results.png")
    print("  • simulation_report.txt\n")


# -----------------------------------------------------------------------------
# GENERACIÓN DE INFORME
# -----------------------------------------------------------------------------
def generate_report(queue_results, theoretical_metrics, markov_results,
                    stationary_dist, bootstrap_results, parameters) -> str:
    """Genera un informe textual completo con los resultados de la simulación."""
    lines = []
    lines.append("=" * 80)
    lines.append("INFORME DE SIMULACIÓN DE RED")
    lines.append("Sistema con Teoría de Colas, Cadenas de Markov y Bootstrap")
    lines.append("=" * 80 + "\n")

    # Parámetros
    lines.append("1. PARÁMETROS DE CONFIGURACIÓN")
    lines.append("-" * 80)
    for k, v in parameters.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    # Resultados del sistema de colas
    lines.append("2. RESULTADOS DEL SISTEMA DE COLAS")
    lines.append("-" * 80)
    lines.append(f"  ρ: {queue_results['rho']:.4f}")
    lines.append(f"  Wq: {queue_results['Wq']:.4f} min")
    lines.append(f"  Ws: {queue_results['Ws']:.4f} min")
    lines.append(f"  Lq: {queue_results['Lq']:.4f}")
    lines.append(f"  Ls: {queue_results['Ls']:.4f}")
    lines.append(f"  Atendidas: {queue_results['served']} | Rechazadas: {queue_results['rejected']}\n")

    if theoretical_metrics:
        lines.append("  Comparación con valores teóricos:")
        for metric in ["rho", "Lq"]:
            err = abs(queue_results[metric] - theoretical_metrics[metric]) / theoretical_metrics[metric] * 100
            lines.append(f"    {metric.upper()}: {theoretical_metrics[metric]:.4f} (Error: {err:.2f}%)")
        lines.append("")

    # Cadenas de Markov
    lines.append("3. CADENA DE MARKOV")
    lines.append("-" * 80)
    for state, prob in zip(["Operativo", "Degradado", "Fallido"], stationary_dist):
        lines.append(f"  {state}: {prob:.4f} ({prob * 100:.2f}%)")
    lines.append("")

    # Bootstrap
    lines.append("4. VALIDACIÓN BOOTSTRAP (95% IC)")
    lines.append("-" * 80)
    for metric, res in bootstrap_results.items():
        cv = res["std"] / res["mean"]
        level = "Robusto" if cv < 0.1 else "Moderado" if cv < 0.2 else "Variable"
        lines.append(f"  {metric}: μ={res['mean']:.4f}, IC95%=[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}], σ={res['std']:.4f}, CV={cv:.4f} ({level})")
    lines.append("")

    # Conclusiones
    lines.append("5. CONCLUSIONES")
    lines.append("-" * 80)
    lines.append(f"  {'Sistema estable' if queue_results['rho'] < 1 else '⚠️ Sistema inestable'}")
    lines.append(f"  El sistema opera correctamente el {stationary_dist[0] * 100:.1f}% del tiempo")
    lines.append(f"  Estado degradado: {stationary_dist[1] * 100:.1f}% | Fallo: {stationary_dist[2] * 100:.1f}%")
    lines.append("\n" + "=" * 80)
    lines.append("FIN DEL INFORME")
    lines.append("=" * 80)

    return "\n".join(lines)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
