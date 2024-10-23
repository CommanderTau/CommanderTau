#include <iostream>
#include <cmath>

using namespace std;

// Функция, уравнение которой мы решаем
double f(double x) {
    return x - 2*sin(x); // х – 2sin(x) = 0

}

// Производная функции f(x)
double f_derivative(double x) {
    return 2 * x; // Производная функции - 2x
}

// Метод Ньютона для решения уравнения f(x) = 0
double newton_method(double x0, double tol, int max_iter) {
    double x = x0;

    for (int iter = 0; iter < max_iter; ++iter) {
        double fx = f(x);
        double f_prime_x = f_derivative(x);

        if (abs(fx) < tol) {
            cout << "Метод Ньютона сходится на " << iter + 1 << " итерации." << endl;
            return x;
        }

        // Обновляем значение x с помощью метода Ньютона
        x = x - fx / f_prime_x;
    }

    cout << "Метод Ньютона не сошелся после " << max_iter << " итераций." << endl;
    return x;
}

int main() {
    setlocale(LC_ALL, "Russian");
    double x0 = 1.0;   // Начальное приближение
    double tol = 1e-6; // Точность
    int max_iter = 100; // Максимальное количество итераций

    double root = newton_method(x0, tol, max_iter);
    cout << "Приближенный корень методом Ньютона: " << root << endl;

    return 0;
}
