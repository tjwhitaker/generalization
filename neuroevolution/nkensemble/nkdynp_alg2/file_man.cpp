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
void file_output(void) {
  int i;
  FILE *Time_file;
  char *name_p;
  char name[CHAR_LEN];

  name_p = name;

  // Time for each run
  sprintf(name, "data/time_algn.dat");
  if ((Time_file = fopen(name_p, "w")) == NULL) {
    puts("The file time to be saved cannot be open \n");
    exit(1);
  }

  fprintf(Time_file, "%f ", time_run);

  fclose(Time_file);
}
