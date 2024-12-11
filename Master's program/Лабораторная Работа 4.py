from scipy.optimize import linprog
import numpy as np
import pandas as pd

c = [-23, -25, -19, -24, -25, -15, -12]
A_eq = [
    [9, -2, -8, 21, -9, 25, 20],
    [30, 17, 3, 18, 8, 29, -9]
]
b_eq = [62, 101]
A_ub = [
    [-13, -30, -1, 4, -2, 6, -16],
    [-12, 1, 4, 4, 5, -5, -11],
    [-5, 30, 9, -5, 16, 13, -8]
]
b_ub = [-43, -7, 56]

result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
print("Прямая задача:")
print("Оптимальное значение целевой функции:", -result.fun)
print("Значения переменных:", result.x)

c_dual = [62, 101, 43, 7, 56]
A_dual = [
    [9, 30, 13, 12, -5],
    [-2, 17, 30, -1, 30],
    [-8, 3, 1, -4, 9],
    [21, 18, -4, -4, -5],
    [-9, 8, 2, -5, 16],
    [25, 29, -6, 5, 13],
    [20, -9, 16, 11, -8]
]
b_dual = [23, 25, 19, 24, 25, 15, 12]

result_dual = linprog(c_dual, A_ub=A_dual, b_ub=b_dual, method='highs')
print("Двойственная задача:")
print("Оптимальное значение целевой функции:", result_dual.fun)
print("Значения переменных двойственной задачи:", result_dual.x)

def sensitivity_analysis(b_eq_ranges, b_ub_ranges):
    sensitivity_results = []
    for b_eq_variation in b_eq_ranges:
        for b_ub_variation in b_ub_ranges:
            for i, b_eq_val in enumerate(b_eq_variation):
                b_eq_new = b_eq.copy()
                b_eq_new[i] = b_eq_val
                
                for j, b_ub_val in enumerate(b_ub_variation):
                    b_ub_new = b_ub.copy()
                    b_ub_new[j] = b_ub_val
                    
                    result = linprog(c, A_ub=A_ub, b_ub=b_ub_new, A_eq=A_eq, b_eq=b_eq_new, method='highs')
                    
                    if result.success:
                        sensitivity_results.append({
                            'b_eq': b_eq_new,
                            'b_ub': b_ub_new,
                            'Objective Value': -result.fun,
                            'Variables': result.x
                        })
    return sensitivity_results

b_eq_ranges = [[b_eq[i] - 10, b_eq[i] + 10] for i in range(len(b_eq))]
b_ub_ranges = [[b_ub[i] - 10, b_ub[i] + 10] for i in range(len(b_ub))]

sensitivity_results = sensitivity_analysis(b_eq_ranges, b_ub_ranges)

sensitivity_df = pd.DataFrame(sensitivity_results)
print("Результаты анализа устойчивости:")
print(sensitivity_df)