#include <iostream>
#include <cmath>

using namespace std;

// Функция, уравнение которой мы решаем
double f(double x) {
    return x - 2*sin(x); // Пример функции - х – 2sin(x)=0

}

// Метод дихотомии для решения уравнения f(x) = 0
double bisection_method(double a, double b, double tol, int max_iter) {
    if (f(a) * f(b) >= 0) {
        cout << "Неправильные начальные значения a и b." << endl;
        return 0;
    }

    double c;
    for (int iter = 0; iter < max_iter; ++iter) {
        c = (a + b) / 2; // Находим середину отрезка

        if (abs(f(c)) < tol) {
            cout << "Метод дихотомии сходится на " << iter + 1 << " итерации." << endl;
            return c;
        }

        // Делим отрезок пополам
        if (f(c) * f(a) < 0) {
            b = c;
        }
        else {
            a = c;
        }
    }

    cout << "Метод дихотомии не сошелся после " << max_iter << " итераций." << endl;
    return c;
}

int main() {
    setlocale(LC_ALL, "Russian");
    double a = 1.0;   // Левый конец начального интервала
    double b = 3.0;   // Правый конец начального интервала
    double tol = 1e-6; // Точность
    int max_iter = 100; // Максимальное количество итераций

    double root = bisection_method(a, b, tol, max_iter);
    cout << "Приближенный корень методом дихотомии: " << root << endl;

    return 0;
}
