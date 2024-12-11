from random import randint, random
from scipy.optimize import linprog
import matplotlib.pyplot as plt

def inverse(array):
    def invert(x):
        return -x if not isinstance(x, list) else [invert(y) for y in x]
    return invert(array)

def transpose(matrix):
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]

def display_problem(coefficients, inequality_matrix, bounds, problem_type):
    def format_term(coeff, index):
        return f"{coeff}*x{index+1}" if coeff != 1 else f"x{index+1}"
    
    print(" + ".join([format_term(coeff, i) for i, coeff in enumerate(coefficients)]) + " -> " + problem_type)
    for row, bound in zip(inequality_matrix, bounds):
        print(" + ".join([format_term(coeff, i) for i, coeff in enumerate(row)]) + f" <= {bound}")

def random_index(probabilities):
    rand_value = random()
    cumulative_probability = 0
    for index, prob in enumerate(probabilities):
        cumulative_probability += prob
        if cumulative_probability >= rand_value:
            return index

def analyze_game(matrix):
    row_mins = [float(min(row)) for row in matrix]
    best_row = row_mins.index(max(row_mins))
    col_maxs = [float(max([matrix[i][j] for i in range(len(matrix))])) for j in range(len(matrix[0]))]
    best_col = col_maxs.index(min(col_maxs))
    return row_mins, col_maxs, best_row, best_col

def solve_linear_program(coefficients, inequality_matrix, bounds):
    solution = linprog(coefficients, A_ub=inequality_matrix, b_ub=bounds, method='highs')
    strategy = [float(x) / abs(float(solution.fun)) for x in solution.x]
    return float(solution.fun), strategy

def run_simulation(matrix, stratA, stratB, fixed_row, fixed_col, iterations):
    random_outcome = sum(float(matrix[randint(0, len(matrix) - 1)][randint(0, len(matrix[0]) - 1)]) for _ in range(iterations)) / iterations
    fixedA_randomB = sum(float(matrix[fixed_row][randint(0, len(matrix[0]) - 1)]) for _ in range(iterations)) / iterations
    randomA_fixedB = sum(float(matrix[randint(0, len(matrix) - 1)][fixed_col]) for _ in range(iterations)) / iterations
    return random_outcome, fixedA_randomB, randomA_fixedB

def plot_distribution(choices, strategy, axis):
    bins = range(len(strategy) + 1)
    axis.hist(choices, bins=bins, density=True, align='left', facecolor='none', edgecolor='C0', linewidth=3)

matrix = [
    [-2, -6, 3],
    [4, 5, -1],
    [-2, -1, 4],
    [0, 1, 3]
]
row_mins, col_maxs, optimal_row, optimal_col = analyze_game(matrix)
print(f"Чистая стратегия A: A{optimal_row + 1}")
print(f"Чистая стратегия B: B{optimal_col + 1}")
game_lower_value, game_upper_value = max(row_mins), min(col_maxs)
print(f"Нижняя цена игры: {float(game_lower_value)}, верхняя: {float(game_upper_value)}")
if game_lower_value != game_upper_value:
    print("Седловой точки нет")
print("\nПрямая задача линейного программирования:")
objective = [1] * len(matrix)
inequalities = [[-matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]
limits = [-1] * len(matrix[0])
opt_game_value, strategyA = solve_linear_program(objective, inequalities, limits)
print(f"Оптимальная цена игры: {1 / float(opt_game_value)}, стратегия A: {[round(x, 3) for x in strategyA]}")
print("\nДвойственная задача:")
dual_objective, dual_inequalities, dual_limits = inverse(limits), inverse(transpose(inequalities)), objective
dual_game_value, strategyB = solve_linear_program(inverse(dual_objective), dual_inequalities, dual_limits)
print(f"Оптимальная цена игры: {1 / float(dual_game_value)}, стратегия B: {[round(x, 3) for x in strategyB]}")
print("\nИмитационное моделирование:")
iterations = 1000
random_result, optimalA_randomB, randomA_optimalB = run_simulation(matrix, strategyA, strategyB, optimal_row, optimal_col, iterations)
print(f"Случайный выбор стратегий: {round(float(random_result), 3)}")
print(f"A - разумный, B - случайный: {round(float(optimalA_randomB), 3)}")

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
plot_histogram(A_choices, strategyA, axs[0])
plot_histogram(B_choices, strategyB, axs[1])
plt.show()