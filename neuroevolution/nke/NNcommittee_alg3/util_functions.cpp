/******************************************************************************\
*								Diverse Functions						 *
\******************************************************************************/
#include "defs.h"
#include <cmath>
#include <cstdlib>

/******************************************************************************\
*								 Dynamic Allocation: Matrix of Integers					 *
\******************************************************************************/
int **aloc_matrixi(int lines, int collums) {
  int i, **Matrix;

  Matrix = new int *[lines];
  for (i = 0; i < lines; i++) {
    Matrix[i] = new int[collums];
  }
  if (!Matrix) {
    cout << "Allocation Error!" << endl;
    exit(1);
  }

  return Matrix;
}

/******************************************************************************\
*								 Dynamic Allocation: Matrix of Doubles					 *
\******************************************************************************/
double **aloc_matrixd(int lines, int collums) {
  int i;
  double **Matrix;

  Matrix = new double *[lines];
  for (i = 0; i < lines; i++) {
    Matrix[i] = new double[collums];
  }
  if (!Matrix) {
    cout << "Allocation Error!" << endl;
    exit(1);
  }

  return Matrix;
}

/******************************************************************************\
*								Dynamic Allocation: Vector of Integers						 *
\******************************************************************************/
int *aloc_vectori(int lines) {
  int *vector;

  vector = new int[lines];
  if (!vector) {
    cout << "Allocation Error!" << endl;
    exit(1);
  }
  return vector;
}
/******************************************************************************\
*								Dynamic Allocation: Vector of Doubles						 *
\******************************************************************************/
double *aloc_vectord(int lines) {
  double *vector;

  vector = new double[lines];
  if (!vector) {
    cout << "Allocation Error!" << endl;
    exit(1);
  }
  return vector;
}

/******************************************************************************\
*								 Dynamic Desallocation: Matrix of Integers					 *
\******************************************************************************/
void desaloc_matrixi(int **Matrix, int lines) {
  int i;

  for (i = 0; i < lines; i++) {
    delete[] Matrix[i];
  }
  delete[] Matrix;
}

/******************************************************************************\
*								 Dynamic Desallocation: Matrix of Doubles				 *
\******************************************************************************/
void desaloc_matrixd(double **Matrix, int lines) {
  int i;

  for (i = 0; i < lines; i++) {
    delete[] Matrix[i];
  }
  delete[] Matrix;
}

/******************************************************************************\
*								 Random Integer between L_range and H_range			 *
\******************************************************************************/
int random_int(int L_range, int H_range) {
  return ((int)((rand() / (RAND_MAX + 1.0)) * (H_range - L_range + 1) + L_range)); // random integer beteween [L_range and H_range]
}

/******************************************************************************\
*								 Random double in [0.0,1.0]			 *
\******************************************************************************/
double random_dou(void) {
  return (rand() / double(RAND_MAX)); //  random double in [0.0, 1.0]:
}

/******************************************************************************\
*		Random Permutation of the  vector of integers v				 *
\******************************************************************************/
void rand_perm(int *inp, int *out, int size) {
  int i, j;

  out[0] = inp[0];
  for (i = 1; i < size; i++) {
    j = random_int(0, i);
    if (i != j)
      out[i] = out[j];
    out[j] = inp[i];
  }
}

/******************************************************************************\
*								 Transform a binary vector with lenght l into an integer 		 *
 *  																		*
\******************************************************************************/
int binvec2dec(int *x, int l) {
  int i, y = 0;

  for (i = 0; i < l; i++)
    y = y + x[i] * ((int)pow(2.0, (double)i));

  return (y);
}

/******************************************************************************\
*								Auxiliar: Transform an integer into a binary vector with lenght l   *
\******************************************************************************/
void dec2binvec_aux(int number, int *x, int index) {
  int remainder;

  if (number <= 1) {
    x[index] = number;
    return;
  }

  remainder = number % 2;
  x[index] = remainder;
  dec2binvec_aux(number >> 1, x, index + 1);
}

/******************************************************************************\
*								 Transform an integer into a binary vector with lenght l  		 *
\******************************************************************************/
void dec2binvec(int number, int *x, int l) {
  int i;

  for (i = 0; i < l; i++)
    x[i] = 0;

  dec2binvec_aux(number, x, 0);
}

/******************************************************************************\
*								 Multiplication of matrices M=AB : double						 		 *
\******************************************************************************/
void multMatrix(double **M, double **A, int l_A, int c_A, double **B, int l_B, int c_B) {
  int i, j, k;

  if (c_A != l_B) {
    cout << "Error - multiplication: size of the matrices!" << endl;
    exit(1);
  }

  for (i = 0; i < l_A; i++) {
    for (j = 0; j < c_B; j++) {
      M[i][j] = 0.0;
      for (k = 0; k < c_A; k++) {
        M[i][j] = M[i][j] + A[i][k] * B[k][j];
      }
    }
  }
}

/******************************************************************************\
*								 Multiplication of matrix by vector y=Ax : double				 		 *
\******************************************************************************/
void multMatrixVect(double *y, double **A, int l_A, int c_A, double *x, int l_x) {
  int i, j;

  if (c_A != l_x) {
    cout << "Error - multiplication: size of matrix or vector!" << endl;
    exit(1);
  }

  for (i = 0; i < l_A; i++) {
    y[i] = 0.0;
    for (j = 0; j < c_A; j++) {
      y[i] = y[i] + A[i][j] * x[j];
    }
  }
}

/******************************************************************************\
*								 Internal product between x and y : double								 		 *
\******************************************************************************/
double multVect(double *x, double *y, int l) {
  double res = 0.0;
  int i;

  for (i = 0; i < l; i++) {
    res += x[i] * y[i];
  }
  return (res);
}

/******************************************************************************\
*								 Euclidean norm of a vector x : double								 		 *
\******************************************************************************/
double normEuc(double *x, int l) {
  double norm = 0.0;
  int i;

  for (i = 0; i < l; i++)
    norm += x[i] * x[i];

  return (sqrt(norm));
}

/******************************************************************************\
*								 Transpose of matrix M : double								 		 *
\******************************************************************************/
void transpose(double **Mt, double **M, int l, int c) {
  int i, j;

  for (i = 0; i < l; i++) {
    for (j = 0; j < c; j++) {
      Mt[j][i] = M[i][j];
    }
  }
}

/******************************************************************************\
*								 Power method: finding the largest eigenvalue of M			 		 *
\******************************************************************************/
double largEig(double **M, int l, int c) {
  int i;
  double b, temp, *x, *y;

  x = aloc_vectord(c);
  y = aloc_vectord(c);

  for (i = 0; i < c; i++)
    x[i] = 1.0;
  b = normEuc(x, c);
  do {
    multMatrixVect(y, M, l, c, x, c); // y=Mx
    temp = b;
    b = normEuc(y, c); // ||y||
    for (i = 0; i < c; i++)
      x[i] = y[i] / b;

  } while (fabs(b - temp) > 0.000001);

  delete[] y;
  delete[] x;

  return b;
}

/******************************************************************************\
*								 Sign Function		 		                    *
\******************************************************************************/
double sign(double x) {
  if (x > 0.0)
    return (1.0);
  else if (x < 0.0)
    return (-1.0);
  else
    return (0.0);
}

/******************************************************************************\
*								 Sign Function		 		                    *
\******************************************************************************/
int sign_int(double x) {
  if (x > 0.0)
    return (1);
  else if (x < 0.0)
    return (-1);
  else
    return (0);
}
