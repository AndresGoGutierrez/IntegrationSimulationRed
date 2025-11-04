"""
Módulo de Análisis Bootstrap
=============================

Implementa métodos de remuestreo bootstrap para validar
la estabilidad y robustez de las métricas calculadas.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Sequence, Dict, Any


class BootstrapAnalyzer:
    """
    Analizador de bootstrap para validación estadística.

    Utiliza remuestreo con reemplazo para estimar intervalos
    de confianza y evaluar la robustez de las métricas.
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int | None = None
    ) -> None:
        """
        Inicializa el analizador bootstrap.

        Args:
            n_bootstrap: Número de muestras bootstrap.
            confidence_level: Nivel de confianza entre 0 y 1.
            random_seed: Semilla para reproducibilidad.
        """
        if not (0 < confidence_level < 1):
            raise ValueError("El nivel de confianza debe estar entre 0 y 1.")

        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = np.random.default_rng(random_seed)

    # -------------------------------------------------------------------------

    def _percentile_ci(self, samples: np.ndarray) -> tuple[float, float]:
        """Calcula intervalo de confianza percentil."""
        alpha = 1 - self.confidence_level
        lower = np.percentile(samples, 100 * alpha / 2)
        upper = np.percentile(samples, 100 * (1 - alpha / 2))
        return lower, upper

    # -------------------------------------------------------------------------

    def bootstrap_metric(
        self,
        data: Sequence[float],
        metric_func: Callable[[np.ndarray], float] = np.mean
    ) -> Dict[str, Any]:
        """
        Realiza análisis bootstrap de una métrica.

        Args:
            data: Datos originales.
            metric_func: Función para calcular la métrica.

        Returns:
            Resultados del análisis bootstrap.
        """
        data = np.asarray(data)
        n = len(data)
        if n == 0:
            raise ValueError("Los datos no pueden estar vacíos.")

        original_metric = metric_func(data)

        # Generar métricas bootstrap
        bootstrap_metrics = np.array([
            metric_func(self.random_state.choice(data, size=n, replace=True))
            for _ in range(self.n_bootstrap)
        ])

        mean = np.mean(bootstrap_metrics)
        std = np.std(bootstrap_metrics, ddof=1)
        ci_lower, ci_upper = self._percentile_ci(bootstrap_metrics)
        bias = mean - original_metric

        return {
            "original": original_metric,
            "mean": mean,
            "std": std,
            "bias": bias,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence_level": self.confidence_level,
            "bootstrap_samples": bootstrap_metrics,
        }

    # -------------------------------------------------------------------------

    def bootstrap_difference(
        self,
        data1: Sequence[float],
        data2: Sequence[float],
        metric_func: Callable[[np.ndarray], float] = np.mean
    ) -> Dict[str, Any]:
        """
        Realiza bootstrap para la diferencia entre dos grupos.

        Args:
            data1: Datos del primer grupo.
            data2: Datos del segundo grupo.
            metric_func: Función para calcular la métrica.

        Returns:
            Resultados del análisis de diferencia.
        """
        data1, data2 = np.asarray(data1), np.asarray(data2)
        if len(data1) == 0 or len(data2) == 0:
            raise ValueError("Los conjuntos de datos no pueden estar vacíos.")

        original_diff = metric_func(data1) - metric_func(data2)

        bootstrap_diffs = np.array([
            metric_func(self.random_state.choice(data1, size=len(data1), replace=True))
            - metric_func(self.random_state.choice(data2, size=len(data2), replace=True))
            for _ in range(self.n_bootstrap)
        ])

        mean_diff = np.mean(bootstrap_diffs)
        std_diff = np.std(bootstrap_diffs, ddof=1)
        ci_lower, ci_upper = self._percentile_ci(bootstrap_diffs)

        # p-valor de dos colas para diferencia respecto a 0
        p_value = min(2 * np.mean(bootstrap_diffs <= 0), 1.0)
        significant = not (ci_lower <= 0 <= ci_upper)

        return {
            "original_diff": original_diff,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "significant": significant,
        }

    # -------------------------------------------------------------------------

    def bootstrap_correlation(
        self,
        x: Sequence[float],
        y: Sequence[float]
    ) -> Dict[str, Any]:
        """
        Calcula intervalo de confianza bootstrap para correlación de Pearson.

        Args:
            x: Primera variable.
            y: Segunda variable.

        Returns:
            Resultados del análisis de correlación.
        """
        x, y = np.asarray(x), np.asarray(y)
        if len(x) != len(y):
            raise ValueError("x e y deben tener la misma longitud.")

        n = len(x)
        if n == 0:
            raise ValueError("Los datos no pueden estar vacíos.")

        original_corr = np.corrcoef(x, y)[0, 1]

        bootstrap_corrs = np.array([
            np.corrcoef(
                x[idx := self.random_state.choice(n, size=n, replace=True)],
                y[idx]
            )[0, 1]
            for _ in range(self.n_bootstrap)
        ])

        mean_corr = np.mean(bootstrap_corrs)
        std_corr = np.std(bootstrap_corrs, ddof=1)
        ci_lower, ci_upper = self._percentile_ci(bootstrap_corrs)

        return {
            "original_corr": original_corr,
            "mean_corr": mean_corr,
            "std_corr": std_corr,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence_level": self.confidence_level,
        }
