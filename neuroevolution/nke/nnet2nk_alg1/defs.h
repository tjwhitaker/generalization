/******************************************************************************\
*								 Definitions							 *
\******************************************************************************/
#include <iostream>
using namespace std;

// Parameters
extern int n_runs_max;               // number of runs
extern int n_inputs_inp;             // number of inputs for the neural network
extern double con_density_inp;       // connectivity density for the reseivoirs
extern double spectral_radius_d_inp; // spectral radius for the reseivois
extern int t_max;                    // number of steps for evaluation of the solution for the double pole problem

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

// file_man.cpp
void file_output(double **F_el, int N, int K, int n_f_cont, int **M_out, int n_run);
