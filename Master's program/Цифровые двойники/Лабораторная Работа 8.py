import json as js
import numpy as np
import time as tm
import random as rnd

def load_parameters(filepath):
    with open(filepath, 'r') as f:
        return js.loads(f.read())

PARAMETERS = load_parameters('help.json')
DAYS = 365
SLEEP_INTERVAL = 1
USE_RANDOMIZATION = False

def get_cost_array():
    src = PARAMETERS['cost']['sources']
    return np.array(src['mod']) if not USE_RANDOMIZATION else np.array(
        list(map(lambda t: rnd.triangular(*t), zip(src['min'], src['max'], src['mod']))))

def compute_delay(category):
    data = PARAMETERS['delay'][category]
    return int(round(data['mod'])) if not USE_RANDOMIZATION else int(round(rnd.triangular(data['min'], data['max'], data['mod'])))

def retrieve_max_delay(category):
    return PARAMETERS['delay'][category]['mod'] if not USE_RANDOMIZATION else PARAMETERS['delay'][category]['max']

def estimate_profit():
    p = PARAMETERS['profit']
    return np.array(p['mod']) if not USE_RANDOMIZATION else np.array(
        [rnd.triangular(x, y, z) for x, y, z in zip(p['min'], p['max'], p['mod'])])

def calculate_prime_cost(s_cost):
    return np.array([
        sum(np.array(item['sources']) * s_cost) +
        item['energy'] * PARAMETERS['cost']['energy'] +
        item['stuff'] * PARAMETERS['cost']['stuff']
        for item in PARAMETERS['production']
    ])

def calculate_sale_price(shift_arr, s_costs, profits):
    ones = np.ones_like(profits)
    margin = profits + ones
    return shift_arr * calculate_prime_cost(s_costs) * margin

SHIFT_COMBINATIONS = [
    np.array([a, b, c, d]) for a in range(4)
    for b in range(4)
    for c in range(4)
    for d in range(4)
]

INITIAL_CASH = 11000
cash_balance = INITIAL_CASH
lowest_cash = INITIAL_CASH
account_debit = 0
account_credit = 0
total_energy = 0

SALARY_TERM = 15
ENERGY_TERM = 30

inventory = np.zeros(3, dtype=int)
active_shifts = np.array([1] * 4)

daily_source_needs = sum(np.array(p['sources']) * active_shifts[i] for i, p in enumerate(PARAMETERS['production']))
daily_product_output = np.array([p['count'] for p in PARAMETERS['production']]) * active_shifts
daily_energy_usage = sum(p['energy'] * active_shifts[i] for i, p in enumerate(PARAMETERS['production']))
total_staff = sum(p['stuff'] for p in PARAMETERS['production'])

print('Дневные расходы:' + '\n\t' + '\n\t'.join([
    f'Сырьё: {daily_source_needs}',
    f'Персонал: {total_staff}',
    f'Энергия: {daily_energy_usage}'
]))

if not USE_RANDOMIZATION:
    base_cost = calculate_prime_cost(get_cost_array())
    sale_price = calculate_sale_price(active_shifts, get_cost_array(), estimate_profit())
    print(f'Себестоимость: {base_cost},\nцена реализации: {sale_price},\nприбыль: {sale_price - base_cost}')

incoming_materials = []
pending_sales = []
recorded_prices = []

for day in range(DAYS):
    tm.sleep(SLEEP_INTERVAL)
    print('-' * 20 + f'\nДень: {day}')

    while pending_sales and pending_sales[0][0] == day:
        _, sale_sum = pending_sales.pop(0)
        print(f'Реализована продукция на сумму: {sale_sum}')
        account_debit -= sale_sum
        cash_balance += sale_sum

    if day == 0 or (incoming_materials and incoming_materials[0][0] == day):
        if incoming_materials:
            _, qty, cost_struct = incoming_materials.pop(0)
            print(f'Получено сырьё: {qty}')
            inventory += qty
            account_credit -= sum(qty * cost_struct)

        supply_delay = compute_delay('supply')
        total_needed = daily_source_needs * (supply_delay + (retrieve_max_delay('supply') if day > 0 else 0))
        order_quantity = np.maximum(total_needed - inventory, 0)
        print(f'Заказано сырьё: {order_quantity}, будет доставлено к {day + supply_delay}')
        cost_structure = get_cost_array()
        recorded_prices += [cost_structure] * int(min(order_quantity // daily_source_needs))
        incoming_materials.append((day + supply_delay, order_quantity, cost_structure))
        payment = sum(order_quantity * cost_structure)
        cash_balance -= payment
        account_credit += payment

    if np.all(daily_source_needs <= inventory):
        sale_delay = compute_delay('sale')
        revenue = sum(calculate_sale_price(active_shifts, recorded_prices.pop(0), estimate_profit()))
        print(f'Произведена продукция: {daily_product_output}, будет реализована к {day + sale_delay} за {revenue}')
        pending_sales.append((day + sale_delay, revenue))
        pending_sales.sort()
        inventory -= daily_source_needs
        total_energy += daily_energy_usage
        account_debit += revenue

    if day != 0 and day % SALARY_TERM == 0:
        salary_expense = SALARY_TERM * total_staff * PARAMETERS['cost']['stuff']
        print(f'Выплачена зарплата на сумму: {salary_expense}')
        cash_balance -= salary_expense

    if day != 0 and day % ENERGY_TERM == 0:
        energy_expense = total_energy * PARAMETERS['cost']['energy']
        print(f'Оплачена энергия на сумму: {energy_expense}')
        cash_balance -= energy_expense
        total_energy = 0

    print(f'Сырья на складе: {inventory}')
    lowest_cash = min(lowest_cash, cash_balance)
    print(f'Дебет: {account_debit}, кредит: {account_credit}, касса: {cash_balance}')

print(f'Достигнутый минимум средств: {lowest_cash}')
print(f'Начальное количество средств: {INITIAL_CASH}, конечное: {cash_balance}, прибыль: {cash_balance - INITIAL_CASH}')
