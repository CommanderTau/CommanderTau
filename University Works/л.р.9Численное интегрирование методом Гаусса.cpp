#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// Функция, которую мы интегрируем
double func(double x) {
    return 4*x + 1; 
}

// Узлы и веса метода Гаусса для интегрирования на интервале [-1, 1] (2 узла)
vector<double> gauss_nodes = { -0.5773502691896257, 0.5773502691896257 };
vector<double> gauss_weights = { 1.0, 1.0 };

// Функция для численного интегрирования методом Гаусса
double gauss_integration(double (*f)(double), double a, double b) {
    double integral = 0.0;

    // Преобразование интервала интегрирования [-1, 1] в [a, b]
    double A = (b - a) / 2.0;
    double B = (a + b) / 2.0;

    // Интегрирование с использованием узлов и весов Гаусса
    for (int i = 0; i < gauss_nodes.size(); ++i) {
        double x = A * gauss_nodes[i] + B;
        integral += gauss_weights[i] * f(x);
    }

    // Масштабирование интеграла в соответствии с новым интервалом
    integral *= A;

    return integral;
}

int main() {
    setlocale(LC_ALL, "Russian");
    double a = -1.0;  // Нижний предел интегрирования
    double b = 3.0;   // Верхний предел интегрирования

    double result = gauss_integration(func, a, b);
    cout << "Результат интегрирования: " << result << endl;

    return 0;
}
