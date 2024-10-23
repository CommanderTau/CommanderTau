#include <iostream>
#include <time.h>
#include <omp.h>
#include "matrix.h"

using namespace std;

const int l = 2000;
const int m = 2000;
const int n = 2000;

int main(int argc, char* argv[])
{

	cout << "l = " << l << endl;
	cout << "m = " << m << endl;
	cout << "n = " << n << endl;

	cout << "Preparing...\n";

	srand((unsigned)time(NULL));

	double** G1 = new double* [l];
	double* G2 = new double[l * m];
	for (int i = 0; i < l; i++)
	{
		G1[i] = new double[m];
		for (int j = 0; j < m; j++)
		{
			G1[i][j] = rand() * 20.0 / RAND_MAX - 10;
			G2[i * m + j] = G1[i][j];
		}
	}

	double** H1 = new double* [m];
	double* H2 = new double[m * n];
	for (int i = 0; i < m; i++)
	{
		H1[i] = new double[n];
		for (int j = 0; j < n; j++)
		{
			H1[i][j] = rand() * 20.0 / RAND_MAX - 10;
			H2[i * n + j] = H1[i][j];
		}
	}

	double** R1 = new double* [l];
	for (int i = 0; i < l; i++)
	{
		R1[i] = new double[n];
	}

	double* R2 = new double[l * n];

	// ===========================================================================

	cout << "Calculating...\n";

	time_t time1 = time(NULL);
//#pragma omp parallel for   
	for (int i = 0; i < l; i++)
//#pragma omp simd
		for (int j = 0; j < n; j++)
			R1[i][j] = 0;

//#pragma omp parallel for   
	for (int i = 0; i < l; i++)
		for (int k = 0; k < m; k++)
//#pragma omp simd 
			for (int j = 0; j < n; j++)
				R1[i][j] += G1[i][k] * H1[k][j];
	cout << "Usual multiplication G[I][J] ~ " << time(NULL) - time1 << " seconds\n";

	// ============================================================================



matrix G2M(G2, l, m);
matrix H2M(H2, m, n);
matrix R2M(R2, l, n);

time1 = time(NULL);

R2M = G2M * H2M;

/*
#pragma omp parallel for simd
	for (int i = 0; i < l * n; i++)
		R2[i] = 0;

#pragma omp parallel for
	for (int i = 0; i < l; i++)
		for (int k = 0; k < m; k++) {
#pragma omp simd 
			for (int j = 0; j < n; j++)
				R2[i * n + j] += G2[i * m + k] * H2[k * n + j];
		}
*/

	cout << "Usual multiplication G[I*C+J] ~ " << time(NULL) - time1 << " seconds\n";

	// ========================================================================

	cout << "Comparing results...\n";

	for (int i = 0; i < l; ++i)
		for (int j = 0; j < n; ++j)
			if (fabs(R1[i][j] - R2[i * n + j]) > 0.0001) {
				cout << "Error 4\n"; break;
			}
	cout << "Completed...\n";

	return 0;
}

/*
#include <iostream>
#include <stdio.h>
#include "matrix.h"
using namespace std;

double	p=0.9,q=0.1;
int		c,n=100;
double a[7*7]={q,p,0,0,0,0,0,
		       q,0,p,0,0,0,0,
			   q,0,0,p,0,0,0,
			   q,0,0,0,p,0,0,
			   0,0,0,0,p,q,0,
			   0,0,0,0,p,0,q,
			   q,0,0,0,p,0,0};
double x[7]   = {1,0,0,0,0,0,0};
matrix A(a,7,7);
matrix X(x,1,7);

int main(void){
  do{
    cout << c << ' ' << X << endl;
    X=X*A;
  }while(c++<n);
}
*/