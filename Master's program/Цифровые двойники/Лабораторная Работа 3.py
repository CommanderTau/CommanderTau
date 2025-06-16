import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Callable, Tuple, Dict
import enum

class AngleType(enum.Enum):
    AZIMUTH = 1
    ELEVATION = 2
@dataclass
class TrajectoryParameters:
    L: float
    H: float
    X0: float
    V: float
    theta_deg: float
    
    @property
    def theta(self) -> float:
        return np.radians(self.theta_deg)
@dataclass
class InertialSystemParameters:
    T1: float
    T2: float
    delta_deg: float
    @property
    def delta(self) -> float:
        return np.radians(self.delta_deg)
    
class TrajectoryCalculator:
    def __init__(self, params: TrajectoryParameters):
        self.params = params
        self._validate_parameters()
    def _validate_parameters(self):
        if self.params.L <= 0 or self.params.H <= 0 or self.params.X0 <= 0:
            raise ValueError("Все параметры должны быть положительными")
        if abs(self.params.theta_deg) >= 90:
            raise ValueError("Угол должен быть между -90 и 90 градусами")
        
    def calculate_initial_position(self) -> Tuple[float, float, float]:
        Y0 = np.sqrt(self.params.L**2 - self.params.X0**2 - self.params.H**2)
        return self.params.X0, Y0, self.params.H
    
    def calculate_velocity_components(self) -> Tuple[float, float]:
        V_y = self.params.V * np.cos(self.params.theta)
        V_z = self.params.V * np.sin(self.params.theta)
        return V_y, V_z

    def compute_trajectory(self, t: np.ndarray) -> Dict[str, np.ndarray]:
        _, Y0, Z0 = self.calculate_initial_position()
        V_y, V_z = self.calculate_velocity_components()
        Y = Y0 - V_y * t
        Z = Z0 - V_z * t
        phi = np.arctan2(Y, self.params.X0)
        psi = np.arctan2(Z, np.sqrt(self.params.X0**2 + Y**2))
        return {
            'time': t,
            'Y': Y,
            'Z': Z,
            'phi': phi,
            'psi': psi
        }

class InertialSystemSimulator:
    def __init__(self, params: InertialSystemParameters):
        self.params = params
    @staticmethod
    def _inertial_model(t: float, y: float, T: float, real_angle_func: Callable) -> float:
        return (real_angle_func(t) - y) / T
        
    def simulate_inertial_system(self, t: np.ndarray, 
                               real_angles: np.ndarray, 
                               angle_type: AngleType) -> np.ndarray:
        T = self.params.T1 if angle_type == AngleType.AZIMUTH else self.params.T2
        angle_func = interp1d(t, real_angles, kind='linear', fill_value="extrapolate")
        sol = solve_ivp(
            lambda t, y: self._inertial_model(t, y, T, angle_func),
            [0, t[-1]],
            [real_angles[0]],
            t_eval=t
        )
        return sol.y[0]
    
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        np.random.seed(42)
        noise = np.random.uniform(-self.params.delta/2, self.params.delta/2, size=len(signal))
        return signal + noise

class Plotter:
    @staticmethod
    def plot_angles(t: np.ndarray, phi: np.ndarray, psi: np.ndarray):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(t, np.degrees(phi), label='Азимут φ(t)', color='blue')
        ax1.set_title('Азимут φ(t)')
        ax1.set_xlabel('Время (с)')
        ax1.set_ylabel('Угол (°)')
        ax1.grid(True)
        ax1.legend()
        ax2.plot(t, np.degrees(psi), label='Возвышение ψ(t)', color='red')
        ax2.set_title('Возвышение ψ(t)')
        ax2.set_xlabel('Время (с)')
        ax2.set_ylabel('Угол (°)')
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_errors(t: np.ndarray, 
                   phi_error: np.ndarray, 
                   psi_error: np.ndarray, 
                   noisy: bool = False):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        suffix = " с помехой" if noisy else ""
        ax1.plot(t, np.degrees(phi_error), label=f'Ошибка азимута{suffix}', color='blue')
        ax1.set_title(f'Ошибка азимута φ(t){suffix}')
        ax1.set_xlabel('Время (с)')
        ax1.set_ylabel('Ошибка (°)')
        ax1.grid(True)
        ax1.legend()
        ax2.plot(t, np.degrees(psi_error), label=f'Ошибка возвышения{suffix}', color='red')
        ax2.set_title(f'Ошибка возвышения ψ(t){suffix}')
        ax2.set_xlabel('Время (с)')
        ax2.set_ylabel('Ошибка (°)')
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        plt.show()

def main():
    traj_params = TrajectoryParameters(
        L=12000,
        H=3000,
        X0=500,
        V=150,
        theta_deg=-15
    )
    inertial_params = InertialSystemParameters(
        T1=0.1,
        T2=0.1,
        delta_deg=0.5
    )
    calculator = TrajectoryCalculator(traj_params)
    _, Y0, _ = calculator.calculate_initial_position()
    V_y, _ = calculator.calculate_velocity_components()
    t_max = Y0 / V_y
    t = np.linspace(0, t_max, 1000)
    trajectory = calculator.compute_trajectory(t)
    Plotter.plot_angles(t, trajectory['phi'], trajectory['psi'])
    simulator = InertialSystemSimulator(inertial_params)
    phi_meas = simulator.simulate_inertial_system(
        t, trajectory['phi'], AngleType.AZIMUTH
    )
    psi_meas = simulator.simulate_inertial_system(
        t, trajectory['psi'], AngleType.ELEVATION
    )
    phi_error = trajectory['phi'] - phi_meas
    psi_error = trajectory['psi'] - psi_meas
    Plotter.plot_errors(t, phi_error, psi_error)
    phi_error_noisy = simulator.add_noise(phi_error)
    psi_error_noisy = simulator.add_noise(psi_error)
    Plotter.plot_errors(t, phi_error_noisy, psi_error_noisy, noisy=True)

if __name__ == "__main__":
    main()