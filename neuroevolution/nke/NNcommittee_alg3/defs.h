/******************************************************************************\
*								 Definitions							 *
\******************************************************************************/
#include <iostream>
using namespace std;

// Parameters
extern int n_runs_max;   // number of runs
extern int n_inputs_inp; // number of inputs for the neural network
extern int t_max;        // number of steps for evaluation of the solution for the double pole problem
extern int t_max_gen;    // number of steps for evaluation of the solution for the double pole problem	- generalization test

/* Function declaration */
// util_functions.cpp
int *aloc_vectori(int lines);
double *aloc_vectord(int lines);
int **aloc_matrixi(int lines, int collums);
double **aloc_matrixd(int lines, int collums);
void desaloc_matrixi(int **Matrix, int lines);
void desaloc_matrixd(double **Matrix, int lines);
int random_int(int L_range, int H_range);
double random_dou(void);
void rand_perm(int *inp, int *out, int size);
int binvec2dec(int *x, int l);
void dec2binvec(int number, int *x, int l);
double largEig(double **M, int l, int c);
double sign(double x);
int sign_int(double x);
// file_man.cpp
void load_sol(int *x, int l, int n_run);
int load_best_output(int *x, int l, int n_run);
void file_output(double *f_com, double *f_esem, double *f_bout, int **res_test);
