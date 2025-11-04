"""
Módulo de Visualización
========================

Proporciona funciones para crear visualizaciones académicas
de los resultados de la simulación.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import expon


class Visualizer:
    """
    Clase para generar visualizaciones de los resultados de simulación.
    """

    def __init__(self, style: str = "whitegrid", palette: str = "husl"):
        """Inicializa el visualizador con configuración de estilo."""
        sns.set_style(style)
        self.colors = sns.color_palette(palette, 8)
        plt.rcParams.update({
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
        })

    # ------------------ MÉTODOS DE VISUALIZACIÓN ------------------ #

    def _init_ax(self, ax=None, figsize=(10, 6)):
        """Crea un eje si no se proporciona uno."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        return ax

    def plot_wait_time_distribution(self, wait_times, ax=None):
        """Grafica la distribución de tiempos de espera en cola."""
        ax = self._init_ax(ax)

        if len(wait_times) == 0:
            ax.text(0.5, 0.5, "Sin datos disponibles", ha="center", va="center")
            return ax

        ax.hist(wait_times, bins=30, density=True, alpha=0.7,
                color=self.colors[0], edgecolor='black', label='Datos simulados')

        mean_wait = np.mean(wait_times)
        if mean_wait > 0:
            rate = 1 / mean_wait
            x = np.linspace(0, np.max(wait_times), 200)
            ax.plot(x, expon.pdf(x, scale=1 / rate), 'r-', lw=2,
                    label=f'Exponencial ajustada (λ={rate:.3f})')

        ax.set(
            xlabel='Tiempo de espera en cola (min)',
            ylabel='Densidad de probabilidad',
            title='Distribución de Tiempos de Espera (Wq)'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_system_time_distribution(self, system_times, ax=None):
        """Grafica la distribución de tiempos totales en el sistema."""
        ax = self._init_ax(ax)

        if len(system_times) == 0:
            ax.text(0.5, 0.5, "Sin datos disponibles", ha="center", va="center")
            return ax

        ax.hist(system_times, bins=30, density=True, alpha=0.7,
                color=self.colors[1], edgecolor='black')

        mean_time = np.mean(system_times)
        median_time = np.median(system_times)
        ax.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_time:.2f}')
        ax.axvline(median_time, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median_time:.2f}')

        ax.set(
            xlabel='Tiempo total en sistema (min)',
            ylabel='Densidad de probabilidad',
            title='Distribución de Tiempos en Sistema (Ws)'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_queue_length_evolution(self, queue_lengths, ax=None):
        """Grafica la evolución de la longitud de la cola en el tiempo."""
        ax = self._init_ax(ax, figsize=(12, 6))

        if len(queue_lengths) == 0:
            ax.text(0.5, 0.5, "Sin datos disponibles", ha="center", va="center")
            return ax

        ax.plot(queue_lengths, color=self.colors[2], alpha=0.7, linewidth=1, label='Longitud de cola')

        if len(queue_lengths) > 50:
            window = 50
            moving_avg = np.convolve(queue_lengths, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(queue_lengths)), moving_avg,
                    color='red', linewidth=2, label=f'Media móvil ({window} eventos)')

        ax.set(
            xlabel='Evento',
            ylabel='Longitud de cola',
            title='Evolución de la Longitud de Cola'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_markov_evolution(self, state_sequence, state_names, ax=None):
        """Grafica la evolución de estados de la cadena de Markov."""
        ax = self._init_ax(ax, figsize=(12, 6))

        if len(state_sequence) == 0:
            ax.text(0.5, 0.5, "Sin datos disponibles", ha="center", va="center")
            return ax

        ax.plot(state_sequence, marker='o', markersize=2, linewidth=0.5, color=self.colors[3])
        ax.set(
            xlabel='Paso de tiempo',
            ylabel='Estado',
            title='Evolución de Estados (Cadena de Markov)'
        )
        ax.set_yticks(range(len(state_names)))
        ax.set_yticklabels(state_names)
        ax.grid(True, alpha=0.3)
        return ax

    def plot_stationary_distribution(self, stationary_dist, observed_freq, state_names, ax=None):
        """Compara distribución estacionaria teórica con frecuencias observadas."""
        ax = self._init_ax(ax)

        x = np.arange(len(state_names))
        width = 0.35

        ax.bar(x - width / 2, stationary_dist, width, label='Teórica (estacionaria)',
               color=self.colors[4], alpha=0.8)
        ax.bar(x + width / 2, observed_freq, width, label='Observada (simulación)',
               color=self.colors[5], alpha=0.8)

        ax.set(
            xlabel='Estado',
            ylabel='Probabilidad',
            title='Distribución Estacionaria vs Observada'
        )
        ax.set_xticks(x)
        ax.set_xticklabels(state_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        return ax

    def plot_transition_matrix(self, transition_matrix, state_names, ax=None):
        """Visualiza la matriz de transición como heatmap."""
        ax = self._init_ax(ax, figsize=(8, 6))

        im = ax.imshow(transition_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(state_names)))
        ax.set_yticks(np.arange(len(state_names)))
        ax.set_xticklabels(state_names, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(state_names)

        for i in range(len(state_names)):
            for j in range(len(state_names)):
                ax.text(j, i, f'{transition_matrix[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=9)

        ax.set(
            title='Matriz de Transición de Estados',
            xlabel='Estado destino',
            ylabel='Estado origen'
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Probabilidad', rotation=270, labelpad=20)
        return ax

    def plot_bootstrap_distribution(self, bootstrap_samples, ci_lower, ci_upper, metric_name, ax=None):
        """Grafica la distribución bootstrap con intervalo de confianza."""
        ax = self._init_ax(ax)

        if len(bootstrap_samples) == 0:
            ax.text(0.5, 0.5, "Sin datos disponibles", ha="center", va="center")
            return ax

        ax.hist(bootstrap_samples, bins=50, density=True, alpha=0.7,
                color=self.colors[6], edgecolor='black')

        ax.axvline(ci_lower, color='red', linestyle='--', linewidth=2)
        ax.axvline(ci_upper, color='red', linestyle='--', linewidth=2,
                   label=f'IC 95%: [{ci_lower:.4f}, {ci_upper:.4f}]')

        mean_val = np.mean(bootstrap_samples)
        ax.axvline(mean_val, color='green', linestyle='-', linewidth=2,
                   label=f'Media: {mean_val:.4f}')

        ax.set(
            xlabel=metric_name,
            ylabel='Densidad',
            title=f'Distribución Bootstrap - {metric_name}'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_theoretical_vs_simulated(self, theoretical, simulated, ax=None):
        """Compara métricas teóricas vs simuladas."""
        ax = self._init_ax(ax)

        metrics = ['rho', 'Lq', 'Wq']
        theoretical_vals = [theoretical.get(m, np.nan) for m in metrics]
        simulated_vals = [simulated.get(m, np.nan) for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width / 2, theoretical_vals, width, label='Teórico',
               color=self.colors[0], alpha=0.8)
        ax.bar(x + width / 2, simulated_vals, width, label='Simulado',
               color=self.colors[1], alpha=0.8)

        ax.set(
            xlabel='Métrica',
            ylabel='Valor',
            title='Comparación: Teórico vs Simulado'
        )
        ax.set_xticks(x)
        ax.set_xticklabels(['ρ (Utilización)', 'Lq (Long. cola)', 'Wq (Tiempo cola)'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        return ax
