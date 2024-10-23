#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// Функции системы уравнений
double f1(double x1, double x2, double x3, double x4) {
    return -94*x1 + 17*x2 + 34*x3 + 16*x4 + 243;
}

double f2(double x1, double x2, double x3, double x4) {
    return 32*x1 - 77*x2 -35*x4 -112;
}

double f3(double x1, double x2, double x3, double x4) {
    return 16*x1 - 8*x2 - 100*x3 -12*x4 +43;
}

double f4(double x1, double x2, double x3, double x4) {
    return 9*x1 +21*x2 -13*x3 -100*x4 +83;
}

// Итерационная функция МПИ
void iteration_function(vector<double>& x) {
    double x1_new = (17 * x[1] + 34 * x[2] + 16 * x[3] + 243)/94;
    double x2_new = (32 * x[0] - 35 * x[3] - 112)/77;
    double x3_new = (16 * x[0] - 8 * x[1] - 12 * x[3] + 43)/100;
    double x4_new = (9 * x[0] + 21 * x[1] - 13 * x[2] + 83)/100;

    x[0] = x1_new;
    x[1] = x2_new;
    x[2] = x3_new;
    x[3] = x4_new;
}

// Метод простых итераций для решения системы нелинейных уравнений
void simple_iteration_method(vector<double>& x, double tolerance) {
    int iter = 0;
    vector<double> prev_x = x;
    do {
        iteration_function(x);
        iter++;
    } while (++iter <= 10000 && abs(x[0] - prev_x[0]) > tolerance && abs(x[1] - prev_x[1]) > tolerance &&
        abs(x[2] - prev_x[2]) > tolerance && abs(x[3] - prev_x[3]) > tolerance);

    cout << "Решение найдено после " << iter << " итераций:" << endl;
    cout << "x1 = " << x[0] << ", x2 = " << x[1] << ", x3 = " << x[2] << ", x4 = " << x[3] << endl;
}

int main() {
    setlocale(LC_ALL, "Russian");

    vector<double> x = { 0.0, 0.0, 0.0, 0.0 }; // Начальное приближение
    double tolerance = 0.001; // Точность

    simple_iteration_method(x, tolerance);

    return 0;
}
