#include <iostream>
#include <sstream>
#include <time.h>
#include <omp.h>
#include "matrix.h"

using namespace std;

int main(int argc, char* argv[])
{
	int m, n;

	cout << "Input matrix dimensions...\n";

	cout << "m = "; cin >> m;
	cout << "n = "; cin >> n;

	matrix A(m, n);
	matrix x(n, 1);
	matrix b(n, 1);

	x(0) = 1;
	x(1) = 1;

	b(0) = 3;
	b(1) = 7;

	cin >> A;
	
	x = b;
	solve(A, x);

	cout << "m = " << m << "n = " << n << endl;
	cout << A << endl;
	cout << b << endl;
	cout << x << endl;
}
