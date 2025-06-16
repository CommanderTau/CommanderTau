import pandas as pd
import random
import matplotlib.pyplot as plt

K = 7500  
C1 = 160  
P = 15    
O = 8000  
M1 = 3   
C2 = 240  
M2 = 2.5    
cost = O * C1 * 1.3
interest = K * (P / 100) / 12 * M1
income = O * C2
months = ['Месяц 1', 'Месяц 2', 'Месяц 3', 'Месяц 4']
cash_flow = [-cost, -interest * 1000, 0, income]
df_plan = pd.DataFrame({
    'Месяц': months,
    'Денежный поток (руб.)': cash_flow
})
total_cost = cost + interest * 1000  
payback_period = total_cost / (income / M2)  
M1_fact = 4  
C2_fact = 220  
interest_fact = K * (P / 100) / 12 * M1_fact
income_fact = O * C2_fact
months_fact = ['Месяц 1', 'Месяц 2', 'Месяц 3', 'Месяц 4', 'Месяц 5']
cash_flow_fact = [-cost, -interest_fact * 1000, 0, 0, income_fact]  
df_fact = pd.DataFrame({
    'Месяц': months_fact,
    'Денежный поток (руб.)': cash_flow_fact
})
total_cost_fact = cost + interest_fact * 1000  
payback_period_fact = total_cost_fact / (income_fact / M2)  
probabilities = [0.5, 0.2, 0.2, 0.1]
M2_options = [2.5, 3.5, 4.5, 5.5]  

def random_M2():
    return random.choices(M2_options, probabilities)[0]

scenarios = []
for i in range(10):
    M2_random = random_M2()
    interest_random = K * (P / 100) / 12 * (M1_fact + M2_random)  
    income_random = O * C2_fact
    total_cost_random = cost + interest_random * 1000
    payback_period_random = total_cost_random / (income_random / M2_random)

    scenarios.append({
        'Сценарий': i + 1,
        'Срок реализации (мес.)': M2_random,
        'Проценты по кредиту (тыс. руб.)': interest_random,
        'Доход (руб.)': income_random,
        'Срок окупаемости (мес.)': payback_period_random
    })
df_scenarios = pd.DataFrame(scenarios)
average_payback = df_scenarios['Срок окупаемости (мес.)'].mean()

print(f"Затраты на закупку и пошлину: {cost} руб.")
print(f"Проценты по кредиту за {M1} месяцев: {interest} тыс. руб.")
print(f"Доход от реализации: {income} руб.")
print("\nФинансовое состояние по плану:")
print(df_plan)
print(f"\nСрок окупаемости: {payback_period:.2f} месяцев")
print(f"Проценты по кредиту за {M1_fact} месяцев: {interest_fact} тыс. руб.")
print(f"Доход от реализации: {income_fact} руб.")
print("\nФинансовое состояние с учетом изменений:")
print(df_fact)
print(f"\nСрок окупаемости с учетом изменений: {payback_period_fact:.2f} месяцев")
print("\nРезультаты моделирования с учетом случайного срока реализации:")
print(df_scenarios)
print(f"\nСредний срок окупаемости: {average_payback:.2f} месяцев")

plt.figure(figsize=(10, 5))
plt.plot(df_plan['Месяц'], df_plan['Денежный поток (руб.)'], label='План', marker='o')
plt.plot(df_fact['Месяц'], df_fact['Денежный поток (руб.)'], label='Факт', marker='x')
plt.title('Финансовое состояние бизнеса по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Денежный поток (руб.)')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 5))
plt.bar(df_scenarios['Сценарий'], df_scenarios['Срок окупаемости (мес.)'])
plt.title('Срок окупаемости для случайных сценариев')
plt.xlabel('Сценарий')
plt.ylabel('Срок окупаемости (мес.)')
plt.grid(True)
plt.show()
