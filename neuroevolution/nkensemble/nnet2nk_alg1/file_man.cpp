/******************************************************************************\
*				  				 Files Manipulation							 *
\******************************************************************************/

#include "defs.h"
#include <stdio.h>
#include <stdlib.h>

#define CHAR_LEN 1000

/******************************************************************************\
* 										Save data : end of the simulation						 *
\******************************************************************************/
void file_output(double **F_el, int N, int K, int n_f_cont, int **M_out, int n_run) {
  int i, j;
  FILE *NK_file;
  char *name_p;
  char name[CHAR_LEN];

  name_p = name;

  // Save Component Functions F_el
  sprintf(name, "data/NK_%d.dat", n_run);

  if ((NK_file = fopen(name_p, "w")) == NULL) {
    puts("The file NK to be saved cannot be open \n");
    exit(1);
  }

  fprintf(NK_file, "Parameters\n");
  fprintf(NK_file, "%d %d\n", N, K);
  fprintf(NK_file, "F_el\n");

  for (i = 0; i < N; i++) {
    for (j = 0; j < n_f_cont; j++) {
      fprintf(NK_file, "%1.14f ", F_el[i][j]);
    }
    fprintf(NK_file, "\n");
  }

  fprintf(NK_file, "M_out\n");

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      fprintf(NK_file, "%d ", M_out[i][j]);
    }
    fprintf(NK_file, "\n");
  }
  fclose(NK_file);
}

/******************************************************************************\
* 										Save data : end of the simulation						 *
\******************************************************************************/
/*void file_output(double **F_el, int N, int K, int n_f_cont, int **M_out, int n_run)
{
	int i, j;
	FILE *Fel_file, *Mou_file, *NKd_file;
	char *name_p;
	char name[CHAR_LEN];

    name_p = name;

    // Save Component Functions F_el
	sprintf(name,"inp_data/Fel_%d.dat",n_run);
	if ((Fel_file = fopen(name_p,"w"))==NULL) {
		puts("The file Fel to be saved cannot be open \n");
		exit(1);
	}
	for (i=0;i<N;i++){
	      for (j=0;j<n_f_cont;j++)
		        fprintf(Fel_file,"%f ",F_el[i][j]);
          fprintf(Fel_file,"\n");         
	}
	fclose(Fel_file);

    // Save Connectivity Matrix M_out
	sprintf(name,"inp_data/Mou_%d.dat",n_run);
	if ((Mou_file = fopen(name_p,"w"))==NULL) {
		puts("The file Mou to be saved cannot be open \n");
		exit(1);
	}
	for (i=0;i<N;i++){
	      for (j=0;j<N;j++)
		        fprintf(Mou_file,"%d ",M_out[i][j]);
          fprintf(Mou_file,"\n");         
	}
	fclose(Mou_file);

    // Save Component Functions N and K
	sprintf(name,"inp_data/NKd_%d.dat",n_run);
	if ((NKd_file = fopen(name_p,"w"))==NULL) {
		puts("The file NKd to be saved cannot be open \n");
		exit(1);
	}
    fprintf(NKd_file,"%d %d \n",N,K);
	fclose(NKd_file);

}*/
