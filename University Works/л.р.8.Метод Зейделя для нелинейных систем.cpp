#include <iostream>
#include <cmath>

using namespace std;

// Функции системы уравнений
double f1(double x1, double x2) {
    return sin(x2)+2*x1-2;
}

double f2(double x1, double x2) {
    return x2+cos(x1-1)-0.7;
}

// Функции для обновления значений x1 и x2 по методу Зейделя
double update_x1(double x1, double x2) {
    return sin(x2)/2 - 1;
}

double update_x2(double x1, double x2) {
    return cos(x1 - 1) - 0.7;
}

// Метод Зейделя для решения нелинейной системы уравнений
void gauss_seidel_nonlinear(double x1, double x2, int max_iter, double tol) {
    double x1_new, x2_new;
    for (int iter = 0; iter < max_iter; ++iter) {
        x1_new = update_x1(x1, x2); // Обновляем x1
        x2_new = update_x2(x1_new, x2); // Обновляем x2, используя новое значение x1
        if (abs(x1_new - x1) < tol && abs(x2_new - x2) < tol) {
            cout << "Решение найдено:" << endl;
            cout << "x1 = " << x1_new << ", x2 = " << x2_new << endl;
            return;
        }
        x1 = x1_new;
        x2 = x2_new;
    }
    cout << "Решение не найдено после " << max_iter << " итераций." << endl;
}

int main() {
    setlocale(LC_ALL, "Russian");

    double x1_guess = 0.0; // Начальное приближение для x1
    double x2_guess = 0.0; // Начальное приближение для x2
    int max_iter = 1000; // Максимальное количество итераций
    double tol = 1e-6; // Погрешность

    gauss_seidel_nonlinear(x1_guess, x2_guess, max_iter, tol);

    return 0;
}
