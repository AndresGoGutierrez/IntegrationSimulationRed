"""
Módulo de Simulación de Cadenas de Markov
==========================================

Simula cadenas de Markov de tiempo discreto para modelar
transiciones entre estados operativos del sistema.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence, Optional, Dict, Any


class MarkovChainSimulator:
    """
    Simulador de cadena de Markov de tiempo discreto.

    Modela la transición entre estados del sistema
    (por ejemplo: Operativo, Degradado, Fallido).
    """

    def __init__(
        self,
        transition_matrix: Sequence[Sequence[float]],
        state_names: Optional[Sequence[str]] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Inicializa el simulador de cadena de Markov.

        Args:
            transition_matrix: Matriz de transición P[i, j].
            state_names: Nombres de los estados (opcional).
            random_seed: Semilla para reproducibilidad.
        """
        self.transition_matrix = np.array(transition_matrix, dtype=float)
        self.n_states = self.transition_matrix.shape[0]
        self.random_state = np.random.default_rng(random_seed)

        # Validaciones estructurales
        if self.transition_matrix.shape[0] != self.transition_matrix.shape[1]:
            raise ValueError("La matriz de transición debe ser cuadrada.")

        if np.any(self.transition_matrix < 0):
            raise ValueError("La matriz de transición no puede contener valores negativos.")

        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError("Cada fila de la matriz de transición debe sumar 1.")

        # Nombres de los estados
        if state_names is None:
            self.state_names = [f"Estado {i}" for i in range(self.n_states)]
        elif len(state_names) != self.n_states:
            raise ValueError("El número de nombres debe coincidir con el número de estados.")
        else:
            self.state_names = list(state_names)

    # ----------------------------------------------------------------------

    def simulate(
        self,
        n_steps: int,
        initial_state: int = 0,
    ) -> Dict[str, Any]:
        """
        Simula la evolución de la cadena de Markov.

        Args:
            n_steps: Número de pasos a simular.
            initial_state: Índice del estado inicial.

        Returns:
            Resultados de la simulación.
        """
        if not (0 <= initial_state < self.n_states):
            raise ValueError(f"El estado inicial debe estar entre 0 y {self.n_states - 1}.")

        if n_steps <= 0:
            raise ValueError("El número de pasos debe ser mayor que cero.")

        states = np.empty(n_steps, dtype=int)
        states[0] = initial_state

        for t in range(1, n_steps):
            current_state = states[t - 1]
            next_state = self.random_state.choice(self.n_states, p=self.transition_matrix[current_state])
            states[t] = next_state

        unique, counts = np.unique(states, return_counts=True)
        state_counts = {self.state_names[i]: int(c) for i, c in zip(unique, counts)}

        return {
            "state_sequence": states.tolist(),
            "state_counts": state_counts,
            "final_state": self.state_names[states[-1]],
        }

    # ----------------------------------------------------------------------

    def calculate_stationary_distribution(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-10,
    ) -> np.ndarray:
        """
        Calcula la distribución estacionaria π que cumple π = πP.

        Args:
            max_iterations: Máximo número de iteraciones del método de potencia.
            tolerance: Criterio de convergencia.

        Returns:
            Vector de distribución estacionaria.
        """
        pi = np.ones(self.n_states) / self.n_states

        for _ in range(max_iterations):
            pi_next = pi @ self.transition_matrix
            if np.allclose(pi_next, pi, atol=tolerance):
                return pi_next
            pi = pi_next

        # Si no converge, retornar el último valor y advertir
        print(f"⚠️ Advertencia: No se alcanzó convergencia en {max_iterations} iteraciones.")
        return pi

    # ----------------------------------------------------------------------

    def get_transition_probability(self, from_state: int, to_state: int) -> float:
        """
        Obtiene la probabilidad de transición entre dos estados.

        Args:
            from_state: Estado origen.
            to_state: Estado destino.

        Returns:
            Probabilidad de transición.
        """
        if not (0 <= from_state < self.n_states and 0 <= to_state < self.n_states):
            raise ValueError("Los índices de estado están fuera de rango.")
        return float(self.transition_matrix[from_state, to_state])

    # ----------------------------------------------------------------------

    def is_irreducible(self) -> bool:
        """
        Verifica si la cadena de Markov es irreducible.

        Una cadena es irreducible si todos los estados son accesibles entre sí.

        Returns:
            True si la cadena es irreducible.
        """
        # Método: si (P + I)^n tiene todos los elementos > 0, es irreducible
        P = self.transition_matrix + np.eye(self.n_states)
        reachable = np.linalg.matrix_power(P, self.n_states - 1)
        return np.all(reachable > 0)

    # ----------------------------------------------------------------------

    def steady_state_via_eigen(self) -> np.ndarray:
        """
        Calcula la distribución estacionaria mediante descomposición en autovalores.

        Returns:
            Vector de distribución estacionaria normalizado.
        """
        eigvals, eigvecs = np.linalg.eig(self.transition_matrix.T)
        idx = np.argmin(np.abs(eigvals - 1))
        stationary = np.real(eigvecs[:, idx])
        stationary /= stationary.sum()
        return stationary
