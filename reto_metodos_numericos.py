from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple


@dataclass
class SimulationResult:
    t: List[float]
    y: List[float]


@dataclass
class ErrorMetrics:
    mae: float
    rmse: float
    max_abs: float


def exact_logistic(t: float, y0: float, r: float, k: float) -> float:
    """Solucion exacta de dY/dt = r*Y*(1 - Y/K)."""
    return k / (1.0 + ((k - y0) / y0) * math.exp(-r * t))


def euler_step(f: Callable[[float, float], float], t: float, y: float, h: float) -> float:
    return y + h * f(t, y)


def rk2_step(f: Callable[[float, float], float], t: float, y: float, h: float) -> float:
    k1 = f(t, y)
    k2 = f(t + h, y + h * k1)
    return y + 0.5 * h * (k1 + k2)


def rk4_step(f: Callable[[float, float], float], t: float, y: float, h: float) -> float:
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_ode(
    f: Callable[[float, float], float],
    t0: float,
    y0: float,
    tf: float,
    h: float,
    step_method: Callable[[Callable[[float, float], float], float, float, float], float],
) -> SimulationResult:
    n_steps = int(round((tf - t0) / h))
    if n_steps <= 0:
        raise ValueError("El numero de pasos debe ser positivo.")

    h_adjusted = (tf - t0) / n_steps
    t_values = [t0]
    y_values = [y0]

    t = t0
    y = y0
    for _ in range(n_steps):
        y = step_method(f, t, y, h_adjusted)
        t = t + h_adjusted
        t_values.append(t)
        y_values.append(y)

    return SimulationResult(t=t_values, y=y_values)


def error_metrics(y_num: Sequence[float], y_ref: Sequence[float]) -> ErrorMetrics:
    if len(y_num) != len(y_ref):
        raise ValueError("Las series deben tener la misma longitud.")

    errors = [abs(a - b) for a, b in zip(y_num, y_ref)]
    mae = sum(errors) / len(errors)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(y_num, y_ref)) / len(errors))
    max_abs = max(errors)
    return ErrorMetrics(mae=mae, rmse=rmse, max_abs=max_abs)


def fit_logistic_parameters(
    historical_sales: Sequence[float],
    r_values: Sequence[float],
    k_values: Sequence[float],
) -> Tuple[float, float, float]:
    """Busca r y K por rejilla minimizando SSE frente a datos historicos."""
    y0 = historical_sales[0]
    best_sse = float("inf")
    best_r = None
    best_k = None

    for r in r_values:
        for k in k_values:
            if k <= max(historical_sales):
                continue
            sse = 0.0
            for t, y_obs in enumerate(historical_sales):
                y_hat = exact_logistic(float(t), y0, r, k)
                sse += (y_hat - y_obs) ** 2
            if sse < best_sse:
                best_sse = sse
                best_r = r
                best_k = k

    if best_r is None or best_k is None:
        raise RuntimeError("No se pudieron ajustar parametros.")

    return best_r, best_k, best_sse


def benchmark(
    f: Callable[[float, float], float],
    t0: float,
    y0: float,
    tf: float,
    h: float,
    step_method: Callable[[Callable[[float, float], float], float, float, float], float],
    repeats: int,
) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        solve_ode(f, t0, y0, tf, h, step_method)
    elapsed = time.perf_counter() - start
    return (elapsed / repeats) * 1000.0


def estimate_orders(h_values: Sequence[float], errors: Sequence[float]) -> List[float]:
    orders = []
    for i in range(len(h_values) - 1):
        e1 = errors[i]
        e2 = errors[i + 1]
        h1 = h_values[i]
        h2 = h_values[i + 1]
        if e1 <= 0 or e2 <= 0:
            orders.append(float("nan"))
        else:
            p = math.log(e1 / e2) / math.log(h1 / h2)
            orders.append(p)
    return orders


def write_csv(path: str, headers: Sequence[str], rows: Sequence[Sequence[float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> None:
    print("=" * 72)
    print("METODOS NUMERICOS PARA MODELO EMPRESARIAL (VENTAS)")
    print("=" * 72)

    # Datos historicos (miles de unidades vendidas por mes).
    historical_sales = [
        120.0,
        138.0,
        165.0,
        198.0,
        242.0,
        295.0,
        356.0,
        422.0,
        487.0,
        548.0,
        603.0,
        649.0,
        688.0,
    ]

    t0 = 0.0
    tf = float(len(historical_sales) - 1)
    y0 = historical_sales[0]

    # Ajuste simple de parametros del modelo logisitico:
    # dY/dt = r*Y*(1 - Y/K)
    r_grid = [i / 100.0 for i in range(5, 81)]     # 0.05 a 0.80
    k_grid = [float(i) for i in range(700, 2001, 5)]  # capacidad de mercado
    r_hat, k_hat, best_sse = fit_logistic_parameters(historical_sales, r_grid, k_grid)

    print("\n[RETO 1] Ajuste de datos historicos")
    print(f"r estimado = {r_hat:.4f}")
    print(f"K estimado = {k_hat:.2f}")
    print(f"SSE ajuste = {best_sse:.4f}")

    def sales_ode(_: float, y: float) -> float:
        return r_hat * y * (1.0 - y / k_hat)

    methods: Dict[str, Callable[[Callable[[float, float], float], float, float, float], float]] = {
        "Euler": euler_step,
        "RK2": rk2_step,
        "RK4": rk4_step,
    }

    print("\n[RETO 2 y 3] Solucion numerica para la dinamica de ventas")
    h_main = 1.0  # un paso por mes para comparar directamente con historico

    table_rows = []
    for method_name, step in methods.items():
        sim = solve_ode(sales_ode, t0, y0, tf, h_main, step)
        exact = [exact_logistic(t, y0, r_hat, k_hat) for t in sim.t]
        metrics_exact = error_metrics(sim.y, exact)
        metrics_hist = error_metrics(sim.y, historical_sales)

        table_rows.append((method_name, metrics_exact, metrics_hist))

    print("\nComparacion de error (malla mensual h=1.0):")
    print("Metodo    | MAE vs exacta | RMSE vs exacta | MAE vs historico | RMSE vs historico")
    print("-" * 78)
    for method_name, m_exact, m_hist in table_rows:
        print(
            f"{method_name:<9} | {m_exact.mae:>13.6f} | {m_exact.rmse:>14.6f} |"
            f" {m_hist.mae:>15.6f} | {m_hist.rmse:>16.6f}"
        )

    # Exporta trayectorias para facilitar graficas o informe.
    h_export = 0.25
    euler_sim = solve_ode(sales_ode, t0, y0, tf, h_export, euler_step)
    rk2_sim = solve_ode(sales_ode, t0, y0, tf, h_export, rk2_step)
    rk4_sim = solve_ode(sales_ode, t0, y0, tf, h_export, rk4_step)

    traj_rows = []
    for idx, t in enumerate(rk4_sim.t):
        exact_val = exact_logistic(t, y0, r_hat, k_hat)
        traj_rows.append([
            round(t, 6),
            euler_sim.y[idx],
            rk2_sim.y[idx],
            rk4_sim.y[idx],
            exact_val,
        ])

    write_csv(
        "trayectorias_ventas.csv",
        ["t_mes", "euler", "rk2", "rk4", "exacta"],
        traj_rows,
    )

    print("\nArchivo generado: trayectorias_ventas.csv")

    print("\n[RETO 3] Analisis de eficiencia y convergencia")
    h_values = [1.0, 0.5, 0.25, 0.125]
    bench_records = []

    for method_name, step in methods.items():
        final_errors = []
        for h in h_values:
            sim = solve_ode(sales_ode, t0, y0, tf, h, step)
            y_final_exact = exact_logistic(tf, y0, r_hat, k_hat)
            err_final = abs(sim.y[-1] - y_final_exact)
            final_errors.append(err_final)

            # Repeticiones para medir tiempo promedio por corrida.
            repeat_count = 2000 if h >= 0.5 else 1200
            avg_ms = benchmark(sales_ode, t0, y0, tf, h, step, repeats=repeat_count)

            bench_records.append([
                method_name,
                h,
                err_final,
                avg_ms,
            ])

        orders = estimate_orders(h_values, final_errors)
        order_str = ", ".join(f"{p:.3f}" for p in orders)
        print(f"Orden observado {method_name}: {order_str}")

    print("\nBenchmark por metodo y paso:")
    print("Metodo | h      | Error final vs exacta | Tiempo medio (ms)")
    print("-" * 65)
    for method_name, h, err_final, avg_ms in bench_records:
        print(f"{method_name:<6} | {h:<6.3f} | {err_final:>21.8f} | {avg_ms:>15.6f}")

    write_csv(
        "benchmark_metodos.csv",
        ["metodo", "h", "error_final", "tiempo_medio_ms"],
        bench_records,
    )
    print("\nArchivo generado: benchmark_metodos.csv")

    print("\nConclusion rapida:")
    print("1) RK4 logra menor error para los mismos pasos h.")
    print("2) Euler es mas simple, pero necesita h pequeno para buena precision.")
    print("3) RK2 ofrece equilibrio intermedio entre costo y precision.")


if __name__ == "__main__":
    main()
