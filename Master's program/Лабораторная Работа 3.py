from scipy.optimize import minimize
from sympy import symbols, Eq, solve, diff
import numpy as np

def objective(x):
    x1, x2 = x
    return -( -x1**2 - x2**2 + x1 + 8*x2 )
constraints = [
    {'type': 'ineq', 'fun': lambda x: 7 - (x[0] + x[1])},
    {'type': 'ineq', 'fun': lambda x: 5 - x[1]},
    {'type': 'ineq', 'fun': lambda x: x[0]},
    {'type': 'ineq', 'fun': lambda x: x[1]}
]
x0 = [0, 0]
result_numeric = minimize(objective, x0, constraints=constraints)
numeric_solution = {
    'x1': result_numeric.x[0],
    'x2': result_numeric.x[1],
    'F_max': -result_numeric.fun
}
x1, x2 = symbols('x1 x2')
F = -x1**2 - x2**2 + x1 + 8*x2
dF_dx1 = diff(F, x1)
dF_dx2 = diff(F, x2)
stationary_points = solve((dF_dx1, dF_dx2), (x1, x2))
if isinstance(stationary_points, dict):
    stationary_points = [tuple(stationary_points.values())]
elif isinstance(stationary_points, tuple):
    stationary_points = [stationary_points]
valid_stationary_points = []
for point in stationary_points:
    x1_val, x2_val = point
    if x1_val >= 0 and x2_val >= 0 and x1_val + x2_val <= 7 and x2_val <= 5:
        valid_stationary_points.append((x1_val, x2_val, F.subs({x1: x1_val, x2: x2_val})))
boundary_points = [(0, 0), (7, 0), (2, 5), (0, 5)]
F_values_at_points = [(x1_val, x2_val, F.subs({x1: x1_val, x2: x2_val})) for x1_val, x2_val in boundary_points]
all_points = valid_stationary_points + F_values_at_points
max_F_symbolic = max(all_points, key=lambda point: point[2])
F_max_numeric = numeric_solution['F_max']
F_max_symbolic = max_F_symbolic[2].evalf()

print('Численное решение (максимальное значение функции): ', F_max_numeric)
print('Символьное решение (максимальное значение функции): ', F_max_symbolic)