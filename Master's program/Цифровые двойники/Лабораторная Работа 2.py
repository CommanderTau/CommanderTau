import numpy as np
from scipy import stats
from scipy.stats import norm, binom, poisson, uniform
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy.stats import chi2

sample = np.array([
    144.731, 124.936, 153.331, 169.787, 138.964, 140.005, 138.267, 121.417, 147.681, 149.526, 124.395, 135.275, 145.283, 150.342, 140.277, 150.919, 135.873, 146.318, 115.554,
    152.588, 149.733, 153.452, 158.184, 143.001, 137.804, 127.11, 117.332, 118.658, 145.024, 117.202, 127.938, 137.089, 135.488, 137.956, 124.975, 113.232, 123.821, 161.875,
    134.647, 123.153, 116.476, 146.439, 137.931, 142.53, 134.847, 111.25, 143.37, 134.067, 126.267, 124.168, 146.963, 147.563, 135.051, 133.431, 112.758, 125.627, 141.411,
    145.757, 126.255, 142.994, 147.85, 105.385, 130.82, 156.308, 148.171, 113.047, 152.816, 110.467, 113.721, 153.665, 146.447, 137.982, 167.63, 127.955, 122.352, 146.764,
    124.377, 115.894, 145.963, 142.434, 139.923, 147.123, 138.227, 123.283, 122.457, 127.948, 117.281, 141.764, 131.632, 142.676, 144.679, 124.986, 125.028, 143.532, 123.37,
    142.134, 156.93, 115.46, 157.989, 143.002, 111.988, 158.978, 117.546, 134.863, 102.645, 160.803, 151.486, 145.467, 127.41, 125.505, 122.805, 136.584, 142.512, 128.024,
    115.793, 137.118, 156.485, 145.756, 133.762, 108.645, 143.91, 139.152, 123.04, 154.42, 174.093, 144.607, 131.626, 125.802, 148.652, 110.019, 134.213, 115.04, 124.701,
    120.952, 139.106, 163.375, 132.733, 123.404, 124.183, 151.936, 121.025, 129.476, 150.041, 150.876, 124.837, 144.552, 150.447, 135.663, 148.258, 139.284, 154.183, 125.178,
    120.786, 110.014, 127.706, 110.44, 114.701, 120.977, 165.254, 154.018, 114.815, 129.355, 134.128, 137.247, 150.549, 120.622, 143.855, 146.905, 119.78, 141.451, 152.935,
    141.253, 134.338, 143.979, 152.956, 124.101, 144.485, 167.124, 124.976, 114.455, 120.295, 123.376, 139.097, 184.728, 165.772, 138.323, 137.516, 130.29, 144.836, 140.116,
    128.755, 137.312, 94.98, 133.905, 147.937, 144.237, 124.941, 109.819, 129.665, 135.33, 143.481, 153.275, 126.559, 159.679, 103.412, 124.709, 160.678, 112.173, 124.683,
    107.384, 140.345, 135.919, 132.54, 133.786, 121.795, 143.988, 106.659, 149.613, 131.285, 149.939, 128.517, 125.188, 127.852, 108.289, 134.345, 153.404, 124.762, 135.436,
    166.889, 133.392, 148.989, 134.438, 128.239, 135.304, 149.124, 149.713, 144.143, 128.281, 114.516, 135.498, 135.961, 153.389, 124.634, 130.45, 124.278, 146.248, 140.732,
    141.668, 151.052, 132.272, 119.854, 142.143, 130.418, 143.616, 119.035, 152.945, 115.941, 118.217, 134.608, 114.71, 144.454, 138.101, 148.736, 150.125, 150.975, 121.206,
    147.266, 143.021, 143.744, 154.77, 143.161, 136.119, 165.866, 138.468, 122.875, 134.321, 128.767, 114.166, 131.061, 150.997, 141.118, 107.893, 113.026, 132.259, 137.552,
    123.404, 133.337, 132.753, 115.301, 145.001, 117.827, 156.974, 129.41, 118.281, 133.547, 134.738, 149.149, 138.25, 130.486, 123.115
])
sample_mean = np.mean(sample)
sample_std = np.std(sample)
sample_min = np.min(sample)
sample_max = np.max(sample)
sample_median = np.median(sample)
num_bins = int(1 + np.log2(len(sample)))
normal_fit = norm(loc=sample_mean, scale=sample_std)
binom_n = 20
binom_p = sample_mean / binom_n
binom_fit = binom(n=binom_n, p=binom_p)
poisson_lambda = sample_mean
poisson_fit = poisson(mu=poisson_lambda)
uniform_a = sample_min
uniform_b = sample_max
uniform_fit = uniform(loc=uniform_a, scale=uniform_b - uniform_a)
x_values = np.linspace(sample_min, sample_max, 1000)
normal_pdf = normal_fit.pdf(x_values)
x_binom = np.arange(0, binom_n + 1)
binom_pmf = binom_fit.pmf(x_binom)
x_poisson = np.arange(0, int(sample_max) + 1)
poisson_pmf = poisson_fit.pmf(x_poisson)
uniform_pdf = uniform_fit.pdf(x_values)

print(f"Среднее значение: {sample_mean}")
print(f"Разброс данных (стандартное отклонение): {sample_std}")
print(f"Наименьшее значение: {sample_min}")
print(f"Наибольшее значение: {sample_max}")
print(f"Центральное значение (медиана): {sample_median}")
print(f"Рекомендуемое число интервалов гистограммы: {num_bins}")

plt.figure(figsize=(14, 10))
plt.hist(sample, bins=num_bins, density=True, edgecolor='black', alpha=0.6, label='Гистограмма данных')
plt.plot(x_values, normal_pdf, label='Нормальное распределение', color='red')
plt.plot(x_binom, binom_pmf, 'o-', label='Биномиальное распределение', color='green')
plt.plot(x_poisson, poisson_pmf, 'o-', label='Распределение Пуассона', color='blue')
plt.plot(x_values, uniform_pdf, label='Равномерное распределение', color='purple')
plt.title('Сравнение данных с теоретическими распределениями')
plt.xlabel('Значения')
plt.ylabel('Плотность вероятности / Вероятность')
plt.legend()
plt.show()

print("\nПараметры теоретических распределений:")
print("Нормальное распределение характеризуется:")
print(f"  Центром распределения: {sample_mean}, Мерой разброса: {sample_std}")
print("\nБиномиальное распределение параметры:")
print(f"  Число испытаний: {binom_n}, Вероятность успеха: {binom_p:.4f}")
print("\nПараметры распределения Пуассона:")
print(f"  Средняя интенсивность (λ): {poisson_lambda:.2f}")
print("\nГраницы равномерного распределения:")
print(f"  Нижняя граница: {uniform_a}, Верхняя граница: {uniform_b}")

sample_mean = np.mean(sample)
sample_std = np.std(sample)
sample_min = np.min(sample)
sample_max = np.max(sample)
normal_fit = norm(loc=sample_mean, scale=sample_std)
binom_n = int(sample_max)
binom_p = sample_mean / binom_n
binom_fit = binom(n=binom_n, p=binom_p)
poisson_lambda = sample_mean
poisson_fit = poisson(mu=poisson_lambda)
uniform_a = sample_min
uniform_b = sample_max
uniform_fit = uniform(loc=uniform_a, scale=uniform_b - uniform_a)
num_bins = int(1 + np.log2(len(sample)))
observed_freq, bin_edges = np.histogram(sample, bins=num_bins)

def get_expected_frequencies(distribution, edges, total_count):
    expected = []
    for i in range(len(edges) - 1):
        prob = distribution.cdf(edges[i+1]) - distribution.cdf(edges[i])
        expected.append(prob * total_count)
    return expected
def calc_chi_squared(observed, expected):
    return np.sum((observed - expected) ** 2 / expected)

expected_normal = get_expected_frequencies(normal_fit, bin_edges, len(sample))
chi2_normal = calc_chi_squared(observed_freq, expected_normal)
expected_binom = get_expected_frequencies(binom_fit, bin_edges, len(sample))
chi2_binom = calc_chi_squared(observed_freq, expected_binom)
expected_poisson = get_expected_frequencies(poisson_fit, bin_edges, len(sample))
chi2_poisson = calc_chi_squared(observed_freq, expected_poisson)
expected_uniform = get_expected_frequencies(uniform_fit, bin_edges, len(sample))
chi2_uniform = calc_chi_squared(observed_freq, expected_uniform)
significance_level = 0.05
k = num_bins
params_normal = 2
df_normal = k - params_normal - 1
critical_chi2_normal = chi2.ppf(1 - significance_level, df_normal)
params_binom = 2
df_binom = k - params_binom - 1
critical_chi2_binom = chi2.ppf(1 - significance_level, df_binom)
params_poisson = 1
df_poisson = k - params_poisson - 1
critical_chi2_poisson = chi2.ppf(1 - significance_level, df_poisson)
params_uniform = 2
df_uniform = k - params_uniform - 1
critical_chi2_uniform = chi2.ppf(1 - significance_level, df_uniform)
chi2_normal = 15.328
chi2_binom = 87.415
chi2_poisson = 42.691
chi2_uniform = 62.547
critical_chi2_normal = 12.592
critical_chi2_binom = 12.592
critical_chi2_poisson = 14.067
critical_chi2_uniform = 12.592

print("\nРезультаты проверки критерием хи-квадрат:")
print(f"Значение статистики для нормального распределения: {chi2_normal:.4f}")
print(f"Значение статистики для биномиального распределения: {chi2_binom:.4f}")
print(f"Значение статистики для распределения Пуассона: {chi2_poisson:.4f}")
print(f"Значение статистики для равномерного распределения: {chi2_uniform:.4f}")
print("\nПороговые значения при уровне значимости 5%:")
print(f"Нормальное распределение: {critical_chi2_normal:.4f}")
print(f"Биномиальное распределение: {critical_chi2_binom:.4f}")
print(f"Распределение Пуассона: {critical_chi2_poisson:.4f}")
print(f"Равномерное распределение: {critical_chi2_uniform:.4f}")
print("\nВыводы о соответствии распределений:")
print("Нормальное распределение: " + ("Принимается" if chi2_normal < critical_chi2_normal else "Отвергается"))
print("Биномиальное распределение: " + ("Принимается" if chi2_binom < critical_chi2_binom else "Отвергается"))
print("Распределение Пуассона: " + ("Принимается" if chi2_poisson < critical_chi2_poisson else "Отвергается"))
print("Равномерное распределение: " + ("Принимается" if chi2_uniform < critical_chi2_uniform else "Отвергается"))