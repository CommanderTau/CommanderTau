/*
  minimal matrix class
  with supplement routines
  implementations by A.N.Pankratov (pan@impb.ru)
*/

#include "matrix.h"

matrix::matrix(int rownum,int colnum){
  m=rownum;n=colnum;
  p=new type[m*n];
  __mem_chck(p==NULL,"matrix");
  mark=1;
}

matrix::matrix(matrix const& A) {
    m = A.m; n = A.n; p = new type[m * n]; mark = 1;
    #pragma omp simd
    for (int i = 0; i < m * n; i++)
        p[i] = A.p[i];
}

matrix& matrix::operator =(matrix const& A){
    if (p==NULL){
        m = A.m; n = A.n; p = new type[m*n]; mark = 1; 
    }
    __rng_chck(m != A.m || n != A.n, "mat=mat");
    #pragma omp simd
    for (int i = 0; i < m * n; i++)
        p[i] = A.p[i];
    return *this;
}

matrix& matrix::operator =(type a){
    #pragma omp simd
    for (int i = 0; i < m * n; i++)
        p[i] = a;
    return *this;
}

matrix& matrix::operator +=(matrix const& A){
    __rng_chck(m!=A.m || n!=A.n,"mat+=mat");
    #pragma omp simd
    for (int i = 0; i < m * n; i++)
        p[i] += A.p[i];
    return *this;
}

matrix& matrix::operator +=(type a){
    #pragma omp simd
    for (int i = 0; i < m * n; i++)
        p[i] += a;
    return *this;
}

matrix& matrix::operator -=(matrix const& A){
    __rng_chck(m!=A.m || n!=A.n,"mat-=mat");
    #pragma omp simd
    for (int i = 0; i < m * n; i++)
        p[i] -= A.p[i];
    return *this;
}

matrix& matrix::operator -=(type a){
    #pragma omp simd    
    for (int i = 0; i < m * n; i++)
        p[i] -= a;
    return *this;
}

matrix& matrix::operator *=(type a){
    #pragma omp simd
    for (int i = 0; i < m * n; i++)
        p[i] *= a;
    return *this;
}

matrix& matrix::operator /=(type a){
    #pragma omp simd
    for (int i = 0; i < m * n; i++)
        p[i] /= a;
    return *this;
}

matrix matrix::operator +(matrix const& A)const{
    __rng_chck(m!=A.m || n!=A.n,"mat+mat");
    matrix R(m,n);
    #pragma omp simd
    for (int i = 0; i < m * n; i++)
        R.p[i] = p[i] + A.p[i];
    return R;
}

matrix matrix::operator +(type a)const{
    matrix R(m,n);
    #pragma omp simd
    for (int i = 0; i < m * n; i++)
        R.p[i] = p[i] + a;
    return R;
}

matrix matrix::operator -(matrix const& A)const{
    __rng_chck(m != A.m || n != A.n, "mat+mat");
    matrix R(m, n);
    #pragma omp simd
    for (int i = 0; i < m * n; i++)
        R.p[i] = p[i] - A.p[i];
    return R;
}

matrix matrix::operator -(type a)const{
    matrix R(m, n);
    #pragma omp simd
    for (int i = 0; i < m * n; i++)
        R.p[i] = p[i] - a;
    return R;
}

matrix matrix::operator -(void)const{
    matrix R(m,n);
    #pragma omp simd
    for (int i = 0; i < m * n; i++)
        R.p[i] = -p[i];
    return R;
}

matrix matrix::operator *(matrix const& A)const{
    __rng_chck(n!=A.m,"mat*mat");
    matrix R(m,A.n);
    #pragma omp simd
    for (int i = 0; i < m * A.n; i++)
        R.p[i] = 0;
     for (int i = 0; i < m; i++)
        for (int k = 0; k < n; k++)
            #pragma omp simd
            for (int j = 0; j < A.n; j++)
                R.p[i * A.n + j] += p[i * n + k] * A.p[k * A.n + j];
    return R;
}

matrix matrix::operator *(type a)const{
    matrix R(m,n);
    #pragma omp simd
    for (int i = 0; i < m * n; i++)
        R.p[i] = p[i] * a;
    return R;
}

matrix matrix::operator /(type a)const{
    matrix R(m,n);
    #pragma omp simd
    for (int i = 0; i < m * n; i++)
        R.p[i] = p[i] / a;
    return R;
}

bool matrix::operator ==(matrix const& A)const{
    __rng_chck(m!=A.m || n!=A.n,"mat==mat");
    for (int i = 0; i < m * n; i++)
       if (p[i] != A.p[i])
           return false;
    return true;
}

type norm(matrix const& A){
    type a,aa,r=0;
    for (int i=0 ; i<A.m ; i++){
        a=0;
        for (int j=0 ; j<A.n ; j++)
            a+=((aa=A(i,j))>0 ? aa : -aa );
        if (r<a) r=a;
    }
    return r;
}

matrix Co(matrix const& A){
    matrix R(A.n,A.m);
    for (int i=0 ; i<A.m ; i++)
        for (int j=0 ; j<A.n ; j++)
            R(j,i)=A(i,j);
    return R;
}

matrix Lo(matrix const& A){
    __rng_chck(A.m!=A.n,"Lo");
    matrix R(A.m,A.n);R=0;
    for (int i=0 ; i<A.m ; i++)
        for (int j=0 ; j<i ; j++)
            R(i,j)=A(i,j);
    return R;
}

matrix Di (matrix const& A){
    __rng_chck(A.m!=A.n,"Di");
    matrix R(A.m,A.n);R=0;
    for (int i=0 ; i<A.m ; i++)
        R(i,i)=A(i,i);
    return R;
}

matrix Up (matrix const& A){
    __rng_chck(A.m!=A.n,"Up");
    matrix R(A.m,A.n);R=0;
    for (int i=0 ; i<A.m ; i++)
        for (int j=i+1 ; j<A.n ; j++)
            R(i,j)=A(i,j);
    return R;
}

ostream& operator << (ostream& os,matrix const& A){
    for (int i=0 ; i<A.m ; i++){
        for (int j=0 ; j<A.n ; j++)
            os << A(i,j) << ' ';
        os << endl;
    };
    return os;
}

istream& operator >> (istream& is, matrix& A) {
    for (int i = 0; i < A.m * A.n; i++)
        is >> A(i);
    return is;
}

void euler (matrix& X,type& t,type h,RHS F){
    X+=h*(*F)(t+h/2,X+h/2*(*F)(t,X));
    t+=h;
}

void rk (matrix& X,type& t,type h,RHS F){
    matrix S=(*F)(t,X),R=(*F)(t+=h/2,X+h/2*S);
    S+=2*R;
    S+=2*(R=(*F)(t,X+h/2*R));
    S+=(*F)(t+=h/2,X+h*R);
    X+=h/6*S;
}

void merson (matrix& X,type& t,type& h,type d,RHS F){
    matrix X1,X2,X3,X4,X5;
    matrix F0,F2,F3;
    type t1,t2;
    X1=X+(F0=(*F)(t,X))*h/3;
    X2=X+(F0+(*F)(t1=t+h/3,X1))*h/6;
    X3=X+(F0+3*(F2=(*F)(t1,X2)))*h/8;
    X4=X+(F0/2-1.5*F2+2*(F3=(*F)(t+h/2,X3)))*h;
    X5=X+(F0+F3+(*F)(t2=t+h,X4))*h/6;
    type e=norm(X4-X5);
    if (e>d) 
        h/=2; 
    else{
        X=X4; t=t2; 
        if (e<d/50) 
            h*=2; 
    } 
}

void invP (matrix const& P,matrix& R){
    __rng_chck(P.rownum()!=P.colnum() || R.rownum()!=R.colnum() || P.rownum()!=R.colnum(),"invP");
    int n=P.rownum();
    for (int k=0 ; k<n-1 ; k++)
        for (int i=k+1 ; i<n ; i++){
            R(i,k)=-P(i,k);
            for (int j=k+1 ; j<i ; j++)
	            R(i,k)-=P(i,j)*R(j,k);
        }
}

type compact (matrix const& A,matrix& P,matrix& Q){
  __rng_chck(A.rownum()!=A.colnum() || P.rownum()!=P.colnum() ||
	     Q.rownum()!=Q.colnum() ||
	     P.rownum()!=Q.colnum() || A.rownum()!=P.colnum(),"compact");
  type det=1;
  int  i,j,k,n=A.rownum();
  for (i=0 ; i<n ; i++){
    for (j=i ; j<n ; j++){
      Q(i,j)=A(i,j);
      for (k=0 ; k<i ; k++)
	Q(i,j)-=P(i,k)*Q(k,j);
    }
    if ((det*=Q(i,i))==0)
      return 0;
    for (j=i+1 ; j<n ; j++){
      P(j,i)=A(j,i);
      for (k=0 ; k<i ; k++)
	P(j,i)-=P(j,k)*Q(k,i);
      P(j,i)/=Q(i,i);
    }
  }
  return det;
}

void solveP (matrix const& P,matrix& F){
  __rng_chck(P.rownum()!=P.colnum() || F.rownum()!=P.colnum() ,"solveP");
  int n=P.rownum(),m=F.colnum();
  for (int k=0 ; k<m ; k++)
    for (int i=1 ; i<n ; i++)
      for (int j=0 ; j<i ; j++)
	F(i,k)-=P(i,j)*F(j,k);
}

void solveQ (matrix const& Q,matrix& F){
  __rng_chck(Q.rownum()!=Q.colnum() || F.rownum()!=Q.colnum() ,"solveQ");
  int n=Q.rownum(),m=F.colnum();
  for (int k=0 ; k<m ; k++){
    F(n-1,k)/=Q(n-1,n-1);
    for (int i=n-2 ; i>=0 ; i--){
      for (int j=i+1 ; j<n ; j++)
	F(i,k)-=Q(i,j)*F(j,k);
      F(i,k)/=Q(i,i);
    }
  }
}

type solve (matrix const& A,matrix& F){
  int n=F.rownum();
  matrix B(n,n);
  type det=compact(A,B,B);
  if (det==0){
    cerr << A << "det==0 in solve" ;
    exit(3);
  }
  else{
    solveP(B,F);
    solveQ(B,F);
  }
  return det;
}

void spusk(matrix& X0,FUN F){
  int i,j,k,n=X0.rownum();
  matrix F0(n-1,1),F1(n-1,1),A(n-1,n),At(n,n-1),FF;
  type h=0.001;
  for(k=0;k<3;k++){
    FF=F(X0);
    if (FF.colnum()==1){
      F0=FF;
      for(j=0;j<n;j++){
        X0(j,0)+=h;
        F1=F(X0);
        for(i=0;i<(n-1);i++)
          At(j,i)=A(i,j)=(F1(i,0)-F0(i,0))/h;
        X0(j,0)-=h;
      }
    }else if (FF.colnum()==(n+1))
      for(i=0;i<(n-1);i++){
        F0(i,0)=FF(i,0);
        for(j=0;j<n;j++)
          At(j,i)=A(i,j)=FF(i,j+1);
      }
    else{
      cerr << FF << "wrong FUN in curve" ;
      exit(3);
    }
    solve(A*At,F0);
    X0-=At*F0;
  }
}

void curve (matrix& X1,matrix& X2,FUN F,int c){
  matrix X0;
  X0=X2*2-X1;
  spusk(X0,F);
  switch (c) {
    case    3: X1=(X1+X2)/2; break;
    case    2: X2=X0; break;
    case (-1): X1=X0; break;
    default  : X1=X2; X2=X0; break;
  }
}
