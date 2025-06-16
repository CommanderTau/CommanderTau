import math as mth
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from typing import Callable, List, Tuple


class SimulationParameters:
    def __init__(self):
        self._distance = 12000
        self._height = 3000
        self._rocket_speed = 300
        self._target_speed = 150
        self._target_angle = mth.radians(-15)
        self._k = 1
        self._is_limited_tet = False

    @property
    def initial_conditions(self) -> Tuple[float, float]:
        squared_distance = self._distance ** 2
        squared_height = self._height ** 2
        initial_r = mth.sqrt(squared_distance + squared_height)
        initial_angle = mth.atan(self._height / self._distance)
        return initial_r, initial_angle

    @property
    def distance(self) -> int:
        return self._distance

    @property
    def height(self) -> int:
        return self._height

    @property
    def rocket_speed(self) -> int:
        return self._rocket_speed

    @property
    def target_speed(self) -> int:
        return self._target_speed

    @property
    def target_angle(self) -> float:
        return self._target_angle

    @property
    def k(self) -> int:
        return self._k

    @k.setter
    def k(self, value: int):
        self._k = value

    @property
    def is_limited_tet(self) -> bool:
        return self._is_limited_tet

    @is_limited_tet.setter
    def is_limited_tet(self, value: bool):
        self._is_limited_tet = value


class ODEFunctionFactory:
    def __init__(self, params: SimulationParameters):
        self.params = params

    def __call__(self, t: float, y: List[float]) -> List[float]:
        r, phi, tet, *rest = y
        rocket_speed = self.params.rocket_speed
        target_speed = self.params.target_speed
        target_angle = self.params.target_angle

        angle_diff = tet - phi
        cos_angle_diff = mth.cos(angle_diff)
        sin_angle_diff = mth.sin(angle_diff)
        target_angle_plus_phi = target_angle + phi
        cos_target_phi = mth.cos(target_angle_plus_phi)
        sin_target_phi = mth.sin(target_angle_plus_phi)

        derivative_r = -rocket_speed * cos_angle_diff - target_speed * cos_target_phi

        numerator_phi = -rocket_speed * sin_angle_diff + target_speed * sin_target_phi
        denominator_phi = r
        common_derivative = numerator_phi / denominator_phi
        derivative_phi = common_derivative % (2 * mth.pi)
        derivative_tet = common_derivative % (2 * mth.pi)

        if self.params.is_limited_tet and derivative_tet > 0.6:
            derivative_tet = 0.6

        if self.params.k != 1:
            derivative_tet *= self.params.k

        derivative_target_x = -target_speed * mth.cos(target_angle)
        derivative_target_y = target_speed * mth.sin(target_angle)
        derivative_rocket_x = rocket_speed * mth.cos(tet)
        derivative_rocket_y = rocket_speed * mth.sin(tet)

        return [
            derivative_r,
            derivative_phi,
            derivative_tet,
            derivative_target_x,
            derivative_target_y,
            derivative_rocket_x,
            derivative_rocket_y
        ]


class ODESolver:
    def __init__(self, ode_func: Callable, params: SimulationParameters):
        self.ode_func = ode_func
        self.params = params

    def solve(self, y0: List[float], t_max: float, steps: int = 10000) -> object:
        t_eval = np.linspace(0, t_max, steps)
        sol = solve_ivp(self.ode_func, [0, t_max], y0, t_eval=t_eval)
        return sol


class ParameterContext:
    def __init__(self, params: SimulationParameters, k: int = None, limit_tet: bool = None):
        self.params = params
        self.new_k = k
        self.new_limit_tet = limit_tet
        self.original_k = params.k
        self.original_limit = params.is_limited_tet

    def __enter__(self):
        if self.new_k is not None:
            self.params.k = self.new_k
        if self.new_limit_tet is not None:
            self.params.is_limited_tet = self.new_limit_tet

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.params.k = self.original_k
        self.params.is_limited_tet = self.original_limit


def compute_initial_conditions(params: SimulationParameters) -> List[float]:
    initial_r, initial_angle = params.initial_conditions
    return [initial_r, initial_angle, initial_angle, params.distance, params.height, 0, 0]


def determine_stop_index(sol) -> int:
    for idx in range(1, len(sol.y[0])):
        if sol.y[0, idx] < 0 or sol.y[0, idx] > sol.y[0, idx-1]:
            return idx
    return len(sol.y[0]) - 1


def plot_trajectories(sol, stop_idx: int):
    plt.plot(sol.y[3, :stop_idx], sol.y[4, :stop_idx], label="Траектория цели", lw=2)
    plt.plot(sol.y[5, :stop_idx], sol.y[6, :stop_idx], label="Траектория ракеты", lw=2)

    interval = stop_idx // 10
    for i in range(10):
        pos = i * interval
        plt.plot([sol.y[3, pos], sol.y[5, pos]], [sol.y[4, pos], sol.y[6, pos]],
                 'b-.', alpha=0.3, markersize=5, marker='.')

    plt.grid()
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("График сближения ракеты и цели")
    plt.show()


def plot_overload(sol, stop_idx: int):
    delta_theta = [abs(sol.y[2, i+1] - sol.y[2, i]) * 100 for i in range(stop_idx-1)]
    plt.plot(sol.t[:stop_idx-1], delta_theta, label="θ̇")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Радианы")
    plt.title("График изменения угла вектора скорости θ̇ от времени.")
    plt.show()


def execute_simulation_and_plot(params: SimulationParameters, scenario_msg: str = ""):
    ode_factory = ODEFunctionFactory(params)
    solver = ODESolver(ode_factory, params)
    y0 = compute_initial_conditions(params)
    solution = solver.solve(y0, 30)
    stop_index = determine_stop_index(solution)

    print(scenario_msg)
    print(f"Момент сближения t = {solution.t[stop_index]:.2f}")
    print(f"Цель: ({solution.y[3, stop_index]:.2f}, {solution.y[4, stop_index]:.2f})")
    print(f"Ракета: ({solution.y[5, stop_index]:.2f}, {solution.y[6, stop_index]:.2f})\n")

    plot_trajectories(solution, stop_index)
    plot_overload(solution, stop_index)


def main():
    params = SimulationParameters()

    execute_simulation_and_plot(params)

    with ParameterContext(params, limit_tet=True):
        params.is_limited_tet = True
        execute_simulation_and_plot(params, "\nРасчёт с ограничением перегрузки θ̇ = 0.6")

    scenarios = [
        (2, False, "\nМоделирование при k = 2 без ограничения"),
        (3, False, "\nМоделирование при k = 3 без ограничения"),
        (3.5, False, "\nМоделирование при k = 3.5 без ограничения"),
        (2, True, "\nМоделирование при k = 2 с ограничением"),
        (3, True, "\nМоделирование при k = 3 с ограничением"),
        (3.5, True, "\nМоделирование при k = 3.5 с ограничением"),
    ]

    for k_val, limit, msg in scenarios:
        with ParameterContext(params, k=k_val, limit_tet=limit):
            execute_simulation_and_plot(params, msg)


if __name__ == "__main__":
    main()