#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// Функция для вычисления суммы элементов вектора
double vector_sum(const vector<double>& v) {
    double sum = 0.0;
    for (double x : v) {
        sum += x;
    }
    return sum;
}

// Функция для вычисления нового значения переменной x_i в методе Зейделя
double new_x_value(const vector<vector<double>>& A, const vector<double>& b, const vector<double>& x, int i) {
    int n = A.size();
    double sum1 = 0.0, sum2 = 0.0;

    // Суммируем элементы матрицы A * x_k, где k < i
    for (int j = 0; j < i; ++j) {
        sum1 += A[i][j] * x[j];
    }

    // Суммируем элементы матрицы A * x_k, где k > i
    for (int j = i + 1; j < n; ++j) {
        sum2 += A[i][j] * x[j];
    }

    return (b[i] - sum1 - sum2) / A[i][i];
}

// Метод Зейделя для решения системы линейных уравнений
vector<double> gauss_seidel(const vector<vector<double>>& A, const vector<double>& b, int max_iter, double tol) {
    int n = A.size();
    vector<double> x(n, 0.0); // Начальное приближение
    vector<double> x_new(n); // Новые значения переменных
    double diff; // Разница между старыми и новыми значениями

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < n; ++i) {
            // Вычисляем новое значение x_i
            x_new[i] = new_x_value(A, b, x, i);
        }

        // Проверяем условие остановки
        diff = vector_sum(x_new) - vector_sum(x);
        if (abs(diff) < tol) {
            cout << "Сходимость достигнута на " << iter + 1 << " итерации." << endl;
            return x_new;
        }

        x = x_new; // Обновляем значения переменных для следующей итерации
    }

    cout << "Достигнуто максимальное количество итераций." << endl;
    return x_new;
}

int main() {
    setlocale(LC_ALL, "Russian");

    // Пример системы уравнений: 5.4x - 2.4y + 3.8z = 5.5, 
    //                           2.5x + 6.8y - 1.1z = 4.3,
    //                           2.7x - 0.6y + 1.5z = -3.5
    vector<vector<double>> A = { {5.4, -2.4, 3.8}, {2.5, 6.8, -1.1}, {2.7, -0.6, 1.5} };
    vector<double> b = { 5.5, 4.3, -3.5 };
    int max_iter = 100; // Максимальное количество итераций
    double tol = 1e-6; // Погрешность

    vector<double> solution = gauss_seidel(A, b, max_iter, tol);

    // Вывод решения
    cout << "Решение:" << endl;
    for (size_t i = 0; i < solution.size(); ++i) {
        cout << "x[" << i << "] = " << solution[i] << endl;
    }

    return 0;
}
