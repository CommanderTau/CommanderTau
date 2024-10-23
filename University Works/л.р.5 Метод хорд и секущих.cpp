#include <iostream>
#include <cmath>

using namespace std;

// Функция, уравнение которой мы решаем
double f(double x) {
    return sin(x) + x -2; // Пример функции - sin(x) + x -2
}

// Метод хорд для решения уравнения f(x) = 0
double chord_method(double a, double b, double tol, int max_iter) {
    double x0 = a;
    double x1 = b;
    double fx0 = f(x0);
    double fx1 = f(x1);
    double x_next;

    for (int iter = 0; iter < max_iter; ++iter) {
        x_next = (x0 * fx1 - x1 * fx0) / (fx1 - fx0); // Новая точка через хорду
        double fx_next = f(x_next);

        if (abs(fx_next) < tol) {
            cout << "Метод хорд сходится на " << iter + 1 << " итерации." << endl;
            return x_next;
        }

        // Обновляем значения для следующей итерации
        x0 = x1;
        fx0 = fx1;
        x1 = x_next;
        fx1 = fx_next;
    }

    cout << "Метод хорд не сошелся после " << max_iter << " итераций." << endl;
    return x_next;
}

// Метод секущих для решения уравнения f(x) = 0
double secant_method(double a, double b, double tol, int max_iter) {
    double x0 = a;
    double x1 = b;
    double fx0 = f(x0);
    double fx1 = f(x1);
    double x_next;

    for (int iter = 0; iter < max_iter; ++iter) {
        x_next = x1 - (fx1 * (x1 - x0)) / (fx1 - fx0); // Новая точка через секущую
        double fx_next = f(x_next);

        if (abs(fx_next) < tol) {
            cout << "Метод секущих сходится на " << iter + 1 << " итерации." << endl;
            return x_next;
        }

        // Обновляем значения для следующей итерации
        x0 = x1;
        fx0 = fx1;
        x1 = x_next;
        fx1 = fx_next;
    }

    cout << "Метод секущих не сошелся после " << max_iter << " итераций." << endl;
    return x_next;
}

int main() {
    setlocale(LC_ALL, "Russian");

    double a = 1.0;   // Начальное приближение a
    double b = 3.0;   // Начальное приближение b
    double tol = 1e-6; // Точность
    int max_iter = 100; // Максимальное количество итераций

    // Используем метод хорд для нахождения решения
    double root_chord = chord_method(a, b, tol, max_iter);
    cout << "Приближенный корень методом хорд: " << root_chord << endl;

    // Используем метод секущих для нахождения решения
    double root_secant = secant_method(a, b, tol, max_iter);
    cout << "Приближенный корень методом секущих: " << root_secant << endl;

    return 0;
}
