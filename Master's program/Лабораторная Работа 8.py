from math import factorial, prod
from functools import reduce

def calculate_fine(times, costs):
    return reduce(lambda acc, i: acc + costs[i] * sum(times[:i]), range(len(times)), 0)

def weighted_sort(data, weight_func):
    return sorted(data, key=weight_func, reverse=True)

def factorial_series_sum(limit, base):
    return sum(base**i / factorial(i) for i in range(limit + 1))

def generate_table(title, labels, rows):
    print(f'\n{title}:')
    print('\t'.join(labels))
    for row in rows:
        print('\t'.join(map(str, row)))

time = [12, 18, 20, 26, 35, 23, 16, 17, 15]
cost = [9, 11, 12, 17, 15, 13, 19, 11, 7]
initial_fine = calculate_fine(time, cost)
print(f'\nШтраф за обслуживание в исходном порядке: {initial_fine}')
data = [(i+1, t, c) for i, (t, c) in enumerate(zip(time, cost))]
sorted_data = weighted_sort(data, lambda x: x[2] / x[1])
tuples = [(i+1, *p) for i, p in enumerate(zip(time, cost))]
sorted_tuples = sorted(tuples, key=lambda x: x[2]/x[1], reverse=True)
print('\n№\t' + '\t'.join(str(t[0]) for t in sorted_tuples))
print('T\t' + '\t'.join(str(t[1]) for t in sorted_tuples))
print('C\t' + '\t'.join(str(t[2]) for t in sorted_tuples))
print('T/C\t' + '\t'.join(str(round(t[2] / t[1], 3)) for t in sorted_tuples))
sorted_fine = calculate_fine([d[1] for d in sorted_data], [d[2] for d in sorted_data])
print(f'\nШтраф за обслуживание в порядке убывания относительной стоимости: {sorted_fine}')
wait_time, service_time, revenue, maintenance_cost = 9, 8, 7000, 250
lmbda = 1 / wait_time
mu = 1 / service_time
rho = lmbda / mu
print(f'\nИнтенсивность потока заявок: {lmbda}')
print(f'Интенсивность обслуживания: {mu}')
print(f'Нагрузка системы: {rho}')

def calculate_metrics(rho, lmbda, profit, cost):
    idle_prob = 1 / (1 + rho)
    rejection_prob = rho / (1 + rho)
    avg_queue_length = rho**2 / (1 + rho)
    avg_wait_time = avg_queue_length / (lmbda * (1 + rho))
    relative_capacity = 1 - rejection_prob
    absolute_capacity = lmbda * relative_capacity
    net_profit = absolute_capacity * profit  # Это должно быть число
    return idle_prob, rejection_prob, avg_queue_length, avg_wait_time, relative_capacity, absolute_capacity, net_profit

idle, rejection, avg_len, avg_wait, rel_capacity, abs_capacity, profit_result = calculate_metrics(rho, lmbda, revenue, maintenance_cost)
print(f'\nДля одноканальной системы с отказами:')
print(f'Вероятность простоя: {idle}')
print(f'Вероятность отказа: {rejection}')
print(f'Средняя длина очереди: {avg_len}')
print(f'Среднее время ожидания: {avg_wait}')
print(f'Относительная пропускная способность: {rel_capacity}')
print(f'Абсолютная пропускная способность: {abs_capacity}')
print(f'Получено в час: {profit_result}, потрачено: {maintenance_cost}, доход {profit_result / maintenance_cost} руб на 1 рубль затрат')
channels = 3
print(f'\nМногоканальная система, каналов {channels}:')
p0 = 1 / factorial_series_sum(channels, rho)
probabilities = [p0 * rho**i / factorial(i) for i in range(channels + 1)]

def print_probabilities(probs):
    print(f'Предельные вероятности: {[round(p, 3) for p in probs]}')
    print(f'Вероятность отказа: {round(probs[-1], 3)}')
    print(f'Относительная пропускная способность: {round(1 - probs[-1], 3)}')
    print(f'Абсолютная пропускная способность: {round(lmbda * (1 - probs[-1]), 3)}')

print_probabilities(probabilities)
queue_size = 3
print(f'\nМногоканальная система, каналов {channels}, размер очереди {queue_size}:')
denominator = factorial_series_sum(channels, rho) + rho**channels / factorial(channels) * sum((rho / channels)**i for i in range(queue_size + 1))
queue_probs = [
    (rho**i / factorial(i) / denominator if i <= channels else rho**i / (factorial(channels) * channels**(i - channels)) / denominator)
    for i in range(channels + queue_size + 1)
]
print_probabilities(queue_probs)
avg_queue = sum((k - channels) * queue_probs[k] for k in range(channels + 1, channels + queue_size + 1))
free_channels = sum((channels - k) * queue_probs[k] for k in range(channels))
busy_channels = channels - free_channels
print(f'Средняя длина очереди: {avg_queue}')
print(f'Среднее число свободных каналов: {free_channels}')
print(f'Среднее число занятых приборов: {busy_channels}')