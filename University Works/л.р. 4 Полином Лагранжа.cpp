#include <iostream>
#include <cmath>
#include <vector>

// Функция для вычисления значения интерполяционного полинома Лагранжа в точке x
double lagrange_interpolation(double x, const std::vector<double>& x_nodes, const std::vector<double>& y_nodes) {
    double result = 0.0;
    int n = x_nodes.size();

    for (int i = 0; i < n; ++i) {
        double term = y_nodes[i];
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j]);
            }
        }
        result += term;
    }

    return result;
}

int main() {
    setlocale(LC_ALL, "Russian");
    // Узлы интерполяции и значения функции в этих узлах 
    std::vector<double> x_nodes = { 7.0, 8.0, 9.0 };
    std::vector<double> y_nodes;
    for (double x : x_nodes) {
        y_nodes.push_back(std::sin(x)); // Значения функции в узлах интерполяции
    }

    // Точка, в которой мы хотим вычислить полином Лагранжа и функцию f(x)
    double a = 7.5;

    // Вычисляем значения f(x) и полинома Лагранжа в точке a для новой функции
    double f_a = std::log(a)/ (std::log(a)* std::log(a)); // Значение функции в точке a
    double lagrange_a = lagrange_interpolation(a, x_nodes, y_nodes); // Значение полинома Лагранжа в точке a

    // Выводим значения f(x) и полинома Лагранжа в точке a для новой функции
    std::cout << "f(" << a << ") = " << f_a << std::endl;
    std::cout << "Полином Лагранжа в точке " << a << " = " << lagrange_a << std::endl;

    // Оценка погрешности интерполяции в точке a для новой функции
    double interpolation_error = std::abs(f_a - lagrange_a);
    std::cout << "Погрешность интерполяции в точке " << a << " = " << interpolation_error << std::endl;

    return 0;
}
