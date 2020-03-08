/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-  */
/*
 * Partition Crossover for the NK Landscapes
 * Copyright (C) 2014  Renato Tinos <rtinos@ffclrp.usp.br>
 * 
 * nkpx is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * nkpx is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "defs.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
	 * C++ version 0.4 char* style "itoa":
	 * Written by Luk�s Chmela
	 * Released under GPLv3.
	 */
char *itoa(int value, char *result, int base) {
  // check that the base if valid
  if (base < 2 || base > 36) {
    *result = '\0';
    return result;
  }

  char *ptr = result, *ptr1 = result, tmp_char;
  int tmp_value;

  do {
    tmp_value = value;
    value /= base;
    *ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz"[35 + (tmp_value - value * base)];
  } while (value);

  // Apply negative sign
  if (tmp_value < 0)
    *ptr++ = '-';
  *ptr-- = '\0';
  while (ptr1 < ptr) {
    tmp_char = *ptr;
    *ptr-- = *ptr1;
    *ptr1++ = tmp_char;
  }
  return result;
}

/******************************************************************************\
*				  	Main													 *
\******************************************************************************/
int main(void) {
  int n_run;
  char sb[200], sa[200];
  //clock_t time_start;

  //cout << "\n ***** DynP Search ****" << endl;
  //time_start=clock();
  for (n_run = 0; n_run < n_runs_max; n_run++) {

    //cout <<"Run:"<< n_run <<"\n"<< endl;
    strcpy(sb, "java -jar EfficientHillClimbers-0.0.1-SNAPSHOT-r2014-11-26-181215.jar exact-nk dynp data/NK_");
    itoa(n_run, sa, 10);
    strcat(sb, sa);
    strcat(sb, ".dat -> data/solDP_");
    strcat(sb, sa);
    strcat(sb, ".dat");
    system(sb);
  }
  //time_run = double( clock() - time_start ) / (double)CLOCKS_PER_SEC;
  //file_output();					// save data

  //system("pause");
  return 0;
}
