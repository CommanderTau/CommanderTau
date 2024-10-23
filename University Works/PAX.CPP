// case of boundary condition for linear equation
// How I know a circle 3.1416
// y''+p*p*y=0, y(0)=0, y'(0)=1
// y = sin(p*x)
// y(1,p)=0 => p=pi
// 

#include "matrix.h"

type   p=22/7; // Archimedos
type   pi=4*atan(1.0);

matrix F(type x,const matrix& Y){   // dY/dx=F(x,Y)
  matrix R(4,1);
  R(0)=Y(1);                        // y''+p^2y=0
  R(1)=-p*p*Y(0);
  R(2)=Y(3);                        // z''+2py+p^2z=0    z=dy/dp
  R(3)=-2*p*Y(0)-p*p*Y(2);
  return R;
}

int main(void){
  time_t time1 = time(NULL);
  matrix Y(4,1);

  type x,h=0.000001,h1,d=0.000001;
  for (int i = 0; i < 5; i++) {
    x=0;
    Y=0;
    Y(1)=2;
    while ((x+h)<=1)
        merson(Y, x, h, d, &F);
    merson(Y, x, h1 = 1 - x, d, &F);
        
    p-=Y(0)/Y(2);

    cout << "y = " << Y(0) << endl;
    cout << "p = " << p << endl;
    cout << "p-pi = " << p-pi << endl;
  }
  cout << "Time elapsed" << time(NULL) - time1 << " seconds\n";

}
