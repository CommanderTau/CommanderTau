import numpy as np

matrix = [
    [4, 1, 7, 4],
    [3, 4, 8, 3],
    [3, 0, 6, 6],
    [3, 4, 8, 3]
]
def generate_headers(columns_count):
    return '\tП' + '\tП'.join(map(str, range(1, columns_count + 1)))

def display_matrix(data, row_prefix='А'):
    for idx, row in enumerate(data):
        print(f'{row_prefix}{idx + 1}\t' + '\t'.join(map(str, row)))

def compute_criteria(data, func):
    return [func(row) for row in data]

def display_criteria(values, criteria_func):
    best_value = criteria_func(values)
    for idx, value in enumerate(values):
        print(f'А{idx + 1}\t{value}')

def process_criteria(matrix, probabilities, alpha):
    wald_values = compute_criteria(matrix, min)
    print('\nКритерий Вальда')
    display_criteria(wald_values, max)
    laplace_values = compute_criteria(matrix, lambda row: sum(row) / len(row))
    print('\nКритерий Лапласа')
    display_criteria(laplace_values, max)
    hurwitz_values = [alpha * min(row) + (1 - alpha) * max(row) for row in matrix]
    print(f'\nКритерий Гурвица')
    display_criteria(hurwitz_values, max)
    expectation_values = [sum(p * a for p, a in zip(probabilities, row)) for row in matrix]
    print(f'\nМатожидание')
    display_criteria(expectation_values, max)
    max_value = max(map(max, matrix))
    risk_matrix = [[max_value - a for a in row] for row in matrix]
    print('\nМатрица рисков:')
    display_matrix(risk_matrix, row_prefix='Риск А')
    savage_values = compute_criteria(risk_matrix, max)
    print(f'\nКритерий Сэвиджа')
    display_criteria(savage_values, min)
    return wald_values, laplace_values, hurwitz_values, expectation_values, savage_values

def generate_report(matrix, criteria_results):
    print('\n\tП' + '\tП'.join(map(str, range(1, len(matrix[0]) + 1))) + '\tW\tL\tG\tE\tS')
    for idx, row in enumerate(matrix):
        result = f'А{idx + 1}\t' + '\t'.join(map(str, row))
        result += '\t' + '\t'.join([f"{round(v[idx], 2)}" for v in criteria_results[:-1]]) + '\t' + \
                  f"{round(criteria_results[-1][idx], 2)}"
        print(result)

probabilities = [0.1, 0.3, 0.4, 0.2]
alpha = 0.3
criteria_results = process_criteria(matrix, probabilities, alpha)
generate_report(matrix, criteria_results)