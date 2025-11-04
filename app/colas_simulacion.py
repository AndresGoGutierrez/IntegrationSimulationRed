"""
Módulo de Simulación de Teoría de Colas
========================================

Implementa un simulador de sistema de colas M/M/n con capacidad finita.
Calcula métricas de rendimiento tanto simuladas como teóricas.
"""

import heapq
import math
import numpy as np
from scipy.stats import expon


class QueueSimulator:
    """
    Simulador de sistema de colas M/M/n.

    Características:
    - Llegadas según proceso de Poisson (tasa λ)
    - Tiempos de servicio exponenciales (tasa μ)
    - n servidores en paralelo
    - Capacidad máxima K
    """

    def __init__(self, lambda_rate: float, mu_rate: float, n_servers: int, capacity: int):
        """
        Inicializa el simulador de colas.

        Args:
            lambda_rate (float): Tasa de llegada (solicitudes/unidad de tiempo)
            mu_rate (float): Tasa de servicio (solicitudes/unidad de tiempo)
            n_servers (int): Número de servidores
            capacity (int): Capacidad máxima del sistema
        """
        if lambda_rate <= 0 or mu_rate <= 0:
            raise ValueError("Las tasas λ y μ deben ser positivas.")
        if n_servers < 1:
            raise ValueError("Debe haber al menos un servidor.")
        if capacity < n_servers:
            raise ValueError("La capacidad debe ser ≥ al número de servidores.")

        self.lambda_rate = lambda_rate
        self.mu_rate = mu_rate
        self.n_servers = n_servers
        self.capacity = capacity

    # -----------------------------------------------------------
    # MÉTODO PRINCIPAL DE SIMULACIÓN
    # -----------------------------------------------------------

    def simulate(self, simulation_time: float) -> dict:
        """
        Ejecuta la simulación del sistema de colas M/M/n.

        Args:
            simulation_time (float): Tiempo total de simulación.

        Returns:
            dict: Diccionario con métricas y datos de la simulación.
        """

        # Inicialización de variables
        current_time = 0.0
        queue = []
        servers = [None] * self.n_servers

        # Métricas de rendimiento
        wait_times = []
        system_times = []
        queue_length_history = []
        time_history = []
        served_count = 0
        rejected_count = 0

        # Generar eventos de llegada (precomputados)
        t = 0.0
        events = []
        while t < simulation_time:
            t += expon.rvs(scale=1 / self.lambda_rate)
            if t < simulation_time:
                heapq.heappush(events, (t, "arrival", None))

        # Bucle de simulación de eventos
        while events:
            event_time, event_type, event_data = heapq.heappop(events)
            current_time = event_time

            # Registro histórico
            queue_length_history.append(len(queue))
            time_history.append(current_time)

            if event_type == "arrival":
                total_in_system = len(queue) + sum(s is not None for s in servers)

                if total_in_system >= self.capacity:
                    # Sistema lleno → solicitud rechazada
                    rejected_count += 1
                    continue

                # Buscar servidor libre
                free_server = next((i for i, s in enumerate(servers) if s is None), None)

                if free_server is not None:
                    # Asignar directamente
                    service_time = expon.rvs(scale=1 / self.mu_rate)
                    departure_time = current_time + service_time

                    servers[free_server] = {
                        "arrival_time": current_time,
                        "start_service": current_time,
                        "departure_time": departure_time,
                    }

                    heapq.heappush(events, (departure_time, "departure", free_server))
                    wait_times.append(0.0)
                    system_times.append(service_time)
                else:
                    # Encolar
                    queue.append({"arrival_time": current_time})

            elif event_type == "departure":
                server_id = event_data
                request = servers[server_id]

                # Finaliza servicio
                served_count += 1
                servers[server_id] = None

                # Calcular métricas de la solicitud
                wait_time = request["start_service"] - request["arrival_time"]
                system_time = current_time - request["arrival_time"]
                wait_times.append(wait_time)
                system_times.append(system_time)

                # Atender siguiente en cola si existe
                if queue:
                    next_request = queue.pop(0)
                    service_time = expon.rvs(scale=1 / self.mu_rate)
                    departure_time = current_time + service_time

                    servers[server_id] = {
                        "arrival_time": next_request["arrival_time"],
                        "start_service": current_time,
                        "departure_time": departure_time,
                    }

                    heapq.heappush(events, (departure_time, "departure", server_id))

        # -----------------------------
        # Cálculo de métricas finales
        # -----------------------------
        Wq = np.mean(wait_times) if wait_times else 0.0
        Ws = np.mean(system_times) if system_times else 0.0
        Lq = np.mean(queue_length_history) if queue_length_history else 0.0
        rho = self.lambda_rate / (self.n_servers * self.mu_rate)
        Ls = self.lambda_rate * Ws if Ws > 0 else 0.0

        return {
            "rho": rho,
            "Wq": Wq,
            "Ws": Ws,
            "Lq": Lq,
            "Ls": Ls,
            "served": served_count,
            "rejected": rejected_count,
            "wait_times": np.array(wait_times),
            "system_times": np.array(system_times),
            "queue_length_history": np.array(queue_length_history),
            "time_history": np.array(time_history),
        }

    # -----------------------------------------------------------
    # MÉTODOS TEÓRICOS
    # -----------------------------------------------------------

    def calculate_theoretical_metrics(self) -> dict | None:
        """
        Calcula métricas teóricas para sistema M/M/n (Erlang-C).

        Returns:
            dict: Métricas teóricas, o None si el sistema es inestable.
        """
        n = self.n_servers
        lam, mu = self.lambda_rate, self.mu_rate
        rho = lam / (n * mu)

        if rho >= 1:
            return None  # Sistema inestable

        # Cálculo de P0 (probabilidad de sistema vacío)
        sum_term = sum((n * rho) ** k / math.factorial(k) for k in range(n))
        last_term = ((n * rho) ** n / math.factorial(n)) * (1 / (1 - rho))
        P0 = 1 / (sum_term + last_term)

        # Probabilidad de espera (Erlang C)
        C = ((n * rho) ** n / math.factorial(n)) * (1 / (1 - rho)) * P0

        # Métricas teóricas
        Lq = C * rho / (1 - rho)
        Wq = Lq / lam
        Ws = Wq + 1 / mu
        Ls = lam * Ws

        return {
            "rho": rho,
            "P0": P0,
            "C": C,
            "Lq": Lq,
            "Wq": Wq,
            "Ws": Ws,
            "Ls": Ls,
        }
