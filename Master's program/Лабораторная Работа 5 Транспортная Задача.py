import numpy as np
from scipy.optimize import linprog

supply = np.array([50, 100, 150, 150, 100])
demand = np.array([100, 150, 150, 100, 100])
costs = np.array([
    [3, 4, 5, 4, 7],
    [4, 3, 4, 3, 4],
    [5, 4, 3, 4, 5],
    [3, 4, 4, 3, 4],
    [4, 5, 6, 4, 3]
])
total_supply = supply.sum()
total_demand = demand.sum()
if total_supply > total_demand:
    demand = np.append(demand, total_supply - total_demand)
    costs = np.hstack([costs, np.zeros((costs.shape[0], 1))])
elif total_demand > total_supply:
    supply = np.append(supply, total_demand - total_supply)
    costs = np.vstack([costs, np.zeros((1, costs.shape[1]))])
c = costs.flatten()
A_eq = []
for i in range(len(supply)):
    row = np.zeros_like(costs)
    row[i, :] = 1
    A_eq.append(row.flatten())
b_eq = supply
for j in range(len(demand)):
    row = np.zeros_like(costs)
    row[:, j] = 1
    A_eq.append(row.flatten())
b_eq = np.concatenate([supply, demand])
bounds = [(0, None) for _ in range(len(c))]
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
if result.success:
    x = result.x.reshape(costs.shape)
    print("Оптимальный план перевозок:")
    print(x)
    print("\nМинимальная стоимость перевозок:", result.fun)
else:
    print("Не удалось найти оптимальное решение")

def solve_transportation_problem(costs, supply, demand):
    c = costs.flatten()
    A_eq = []
    for i in range(len(supply)):
        row = np.zeros_like(costs)
        row[i, :] = 1
        A_eq.append(row.flatten())
    for j in range(len(demand)):
        row = np.zeros_like(costs)
        row[:, j] = 1
        A_eq.append(row.flatten())
    b_eq = np.concatenate([supply, demand])
    bounds = [(0, None) for _ in range(len(c))]
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return result

np.random.seed(42)
n_simulations = 1000
results = []
for _ in range(n_simulations):
    random_costs = costs + np.random.uniform(-0.1, 0.1, costs.shape) * costs
    result = solve_transportation_problem(random_costs, supply, demand)
    if result.success:
        results.append(result.fun)
lower_bound = np.percentile(results, 2.5)
upper_bound = np.percentile(results, 97.5)
mean_cost = np.mean(results)
print(f"Средняя стоимость перевозок: {mean_cost}")
print(f"Доверительный интервал: ({lower_bound}, {upper_bound})")

def transportation_method_with_potentials(supply, demand, costs):
    distribution = np.zeros_like(costs, dtype=float)
    supply_left = supply.copy()
    demand_left = demand.copy()
    costs_copy = costs.copy().astype(float)
    while np.sum(supply_left) > 0 and np.sum(demand_left) > 0:
        i, j = divmod(np.argmin(costs_copy), costs_copy.shape[1])
        qty = min(supply_left[i], demand_left[j])
        distribution[i, j] = qty
        supply_left[i] -= qty
        demand_left[j] -= qty
        if supply_left[i] == 0:
            costs_copy[i, :] = np.inf
        if demand_left[j] == 0:
            costs_copy[:, j] = np.inf

    def calculate_potentials(distribution, costs):
        u = np.full(costs.shape[0], np.nan)
        v = np.full(costs.shape[1], np.nan)
        u[0] = 0
        for _ in range(costs.size):
            for i in range(costs.shape[0]):
                for j in range(costs.shape[1]):
                    if distribution[i, j] > 0:
                        if np.isnan(v[j]) and not np.isnan(u[i]):
                            v[j] = costs[i, j] - u[i]
                        elif np.isnan(u[i]) and not np.isnan(v[j]):
                            u[i] = costs[i, j] - v[j]
        for i in range(costs.shape[0]):
            for j in range(costs.shape[1]):
                if distribution[i, j] > 0:
                    if np.isnan(u[i]):
                        u[i] = costs[i, j] - v[j]
                    if np.isnan(v[j]):
                        v[j] = costs[i, j] - u[i]
        return u, v

    def calculate_reduced_costs(u, v, costs):
        reduced_costs = np.zeros_like(costs, dtype=float)
        for i in range(costs.shape[0]):
            for j in range(costs.shape[1]):
                reduced_costs[i, j] = costs[i, j] - (u[i] + v[j] if not np.isnan(u[i]) and not np.isnan(v[j]) else 0)
        return reduced_costs

    def check_optimality(reduced_costs):
        return np.all(reduced_costs >= 0)

    u, v = calculate_potentials(distribution, costs)
    print("Потенциалы строк (u):", u)
    print("Потенциалы столбцов (v):", v)
    reduced_costs = calculate_reduced_costs(u, v, costs)
    print("\nМатрица редуцированных затрат:")
    print(reduced_costs)
    if check_optimality(reduced_costs):
        print("Решение оптимально.")
    else:
        print("Решение не оптимально. Требуется улучшение.")

    return distribution

supply = np.array([50, 100, 150, 150, 100])
demand = np.array([100, 150, 150, 100, 100])
costs = np.array([
    [3, 4, 5, 4, 7],
    [4, 3, 4, 3, 4],
    [5, 4, 3, 4, 5],
    [3, 4, 4, 3, 4],
    [4, 5, 6, 4, 3]
])
print("\nМетод Потенциалов с проверкой оптимальности:")
result = transportation_method_with_potentials(supply, demand, costs)
print("\nРаспределение:")
print(result)