/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-  */
/*
 * Committee of N neural networks 
 * for the double pole balancing problem trained using the NK model Problem
 * 
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
#include "nnetCom.h" // Neural Network class
#include <cmath>
#include <cstdlib>

/******************************************************************************\
*				  	Main													 *
\******************************************************************************/
int main(void) {
  int n_run, *x, i_best, i, **file_res_test;
  double *file_best_fitness_com, *file_best_fitness_esem, *file_best_fitness_bout;

  file_best_fitness_com = aloc_vectord(n_runs_max);
  file_best_fitness_esem = aloc_vectord(n_runs_max);
  file_best_fitness_bout = aloc_vectord(n_runs_max);
  file_res_test = aloc_matrixi(625, n_runs_max);

  cout << "\n ***** Committe of Neural Networks ****" << endl;
  for (n_run = 0; n_run < n_runs_max; n_run++) {
    //srand(n_run+1);	// random seed
    cout << "Run:" << n_run << endl;
    nnetCom *esn = new nnetCom(n_run); // Initialization of the Neural Network (NN)
    //esn->print();
    x = aloc_vectori(esn->n_hid2);

    // Evaluating the Committee (with all inputs)
    load_sol(x, esn->n_hid2, n_run);
    esn->findWeights(x);
    file_best_fitness_com[n_run] = esn->evalGen(x, -1, 0); // Evaluation of committee
    for (i = 0; i < 625; i++)
      file_res_test[i][n_run] = esn->res_test[i];

    // Evaluating the Committee with x=1
    /*	for (i=0;i<esn->n_hid2;i++)
			x[i]=1;
		esn->findWeights(x);
	    file_best_fitness_esem[n_run] = esn->evalGen(x,-1,0);	 // Evaluation of committee*/
    // file_best_fitness_esem[n_run] = esn->evalGen(x,-1,1);	 // Evaluation of esemble

    // Evaluating the Committee (with 20  best inputs)
    //load_sol(x,esn->n_hid2,n_run);
    esn->findWeights2(x);
    file_best_fitness_esem[n_run] = esn->evalGen(x, -1, 0); // Evaluation of committee with 20 best inputs

    // Evaluating the best output
    i_best = load_best_output(x, esn->n_hid2, n_run);
    file_best_fitness_bout[n_run] = esn->evalGen(x, i_best, 0); // Evaluation of the best output

    delete esn;
    delete[] x;
  }

  file_output(file_best_fitness_com, file_best_fitness_esem, file_best_fitness_bout, file_res_test);

  delete[] file_best_fitness_com;
  delete[] file_best_fitness_esem;
  delete[] file_best_fitness_bout;
  desaloc_matrixi(file_res_test, 625);

  return 0;
}
