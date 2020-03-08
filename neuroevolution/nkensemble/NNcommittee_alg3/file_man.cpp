/******************************************************************************\
*				  				 Files Manipulation							 *
\******************************************************************************/

#include "defs.h"
#include <cstring>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#define CHAR_LEN 1000

/******************************************************************************\
* 										 Load solution							 *
\******************************************************************************/
/*void load_sol(int *x, int l, int n_run)
{
	int i;
	FILE *s_file;
	char *name_p;
	char name[64];

    name_p = name;
   	sprintf(name,"data/solDP_%d.dat",n_run);
	if ((s_file = fopen(name_p,"r"))==NULL) {
		cout<<"The file sol. cannot be read!"<<endl;
		exit(1);
	}
	for (i=0;i<l;i++) 
			fscanf(s_file,"%d",&x[i]);

	fclose(s_file);

	
}*/

/******************************************************************************\
* 										 Load solution							 *
\******************************************************************************/
void load_sol(int *x, int l, int n_run) {
  int i;
  char line[CHAR_LEN], *keywords, Delimiters[] = ":\n", name[CHAR_LEN], *name_p, str[2];

  name_p = name;
  sprintf(name, "data/solDP_%d.dat", n_run);

  ifstream fin(name_p);
  while ((fin.getline(line, CHAR_LEN - 1))) {
    if (!(keywords = strtok(line, Delimiters)))
      continue;
    if (!strcmp(keywords, "Optimal solution")) {
      keywords = strtok(NULL, line);
      for (i = 0; i < l; i++) {
        str[0] = keywords[i];
        str[1] = '\0';
        x[l - i - 1] = strtol(str, NULL, 10);
      }
    }
  }

  fin.close();
}

/******************************************************************************\
* 										 Load best output								 *
\******************************************************************************/
int load_best_output(int *x, int l, int n_run) {
  int i, j, n_sub, N_input, K_input, i_best, j_best, *sol_reduced, **M_out_input;
  double **F_el_input, F_best = -1.0;
  char line[CHAR_LEN], *keywords, Delimiters[] = " :=\n\t\r\f\v", name[CHAR_LEN], *name_p;

  name_p = name;
  sprintf(name, "data/NK_%d.dat", n_run);

  ifstream fin(name_p);
  while ((fin.getline(line, CHAR_LEN - 1))) {
    if (!(keywords = strtok(line, Delimiters)))
      continue;
    if (!strcmp(keywords, "Parameters")) {
      fin >> N_input;
      fin >> K_input;
      n_sub = (int)pow(2.0, K_input + 1);
      F_el_input = aloc_matrixd(N_input, n_sub);
      M_out_input = aloc_matrixi(N_input, N_input);
    } else if (!strcmp(keywords, "F_el")) {
      for (i = 0; i < N_input; i++) {
        for (j = 0; j < n_sub; j++) {
          fin >> F_el_input[i][j];
          if (F_el_input[i][j] > F_best) {
            F_best = F_el_input[i][j];
            i_best = i;
            j_best = j;
          }
        }
      }
    } else if (!strcmp(keywords, "M_out")) {
      for (i = 0; i < N_input; i++) {
        for (j = 0; j < N_input; j++) {
          fin >> M_out_input[i][j];
        }
      }
    }
  }

  fin.close();

  // Solution
  sol_reduced = aloc_vectori(K_input + 1);
  dec2binvec(j_best, sol_reduced, K_input + 1);
  for (i = 0; i < N_input; i++)
    x[i] = 0;
  j = 0;
  for (i = 0; i < N_input; i++) {
    if (i == i_best || M_out_input[i_best][i] == 1) {
      x[i] = sol_reduced[j];
      j++;
    }
  }

  delete[] sol_reduced;
  desaloc_matrixd(F_el_input, N_input);
  desaloc_matrixi(M_out_input, N_input);

  return (i_best);
}

/******************************************************************************\
* 										Save data : end of the simulation						 *
\******************************************************************************/
void file_output(double *f_com, double *f_esem, double *f_bout, int **res_test) {
  int i, j;
  FILE *Bestfit_file, *Bestfit2_file, *Bestfit3_file, *res_test_file;
  char *name_p;
  char name[CHAR_LEN];

  name_p = name;

  // Committee
  sprintf(name, "data/best_com.dat");
  if ((Bestfit_file = fopen(name_p, "w")) == NULL) {
    cout << "The file best_com to be saved cannot be open" << endl;
    exit(1);
  }
  for (i = 0; i < n_runs_max; i++) {
    fprintf(Bestfit_file, "%1.14f ", f_com[i]);
  }
  fclose(Bestfit_file);

  // Best output
  sprintf(name, "data/best_out.dat");
  if ((Bestfit2_file = fopen(name_p, "w")) == NULL) {
    cout << "The file best_out to be saved cannot be open" << endl;
    exit(1);
  }
  for (i = 0; i < n_runs_max; i++) {
    fprintf(Bestfit2_file, "%1.14f ", f_bout[i]);
  }
  fclose(Bestfit2_file);

  // Committee with 20 best inputs
  sprintf(name, "data/best_co1.dat");
  if ((Bestfit3_file = fopen(name_p, "w")) == NULL) {
    cout << "The file best_co1 to be saved cannot be open" << endl;
    exit(1);
  }
  for (i = 0; i < n_runs_max; i++) {
    fprintf(Bestfit3_file, "%1.14f ", f_esem[i]);
  }
  fclose(Bestfit3_file);

  // Results for each test (Committee with all inputs)
  sprintf(name, "data/res_test.dat");
  if ((res_test_file = fopen(name_p, "w")) == NULL) {
    cout << "The file res_test to be saved cannot be open" << endl;
    exit(1);
  }
  for (j = 0; j < 625; j++)
    for (i = 0; i < n_runs_max; i++)
      fprintf(res_test_file, "%d ", res_test[j][i]);

  fclose(res_test_file);
}
