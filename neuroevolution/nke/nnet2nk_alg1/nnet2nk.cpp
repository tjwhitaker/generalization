/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-  */
/*
 * Transforming the problem of training a set of N neural networks 
 * for the double pole balancing problem to the NK model Problem
 * Obs.: the NN has two layers. The first one is formed by reservoirs.
 *       the first and second layers are equal for all N NNs
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
#include "nnet.h" // Neural Network class
#include <cmath>
#include <cstdlib>

/******************************************************************************\
*				  	Define M_out (connectivity)			 					 *
\******************************************************************************/
void defConn(int **M_out, int prob_type, int N, int K) {
  int i, j, k, rand_aux, aux, *v_aux, L_aux, H_aux;

  // Initialize output matrix
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      if (i == j) {
        M_out[i][j] = 1;
      } else {
        M_out[i][j] = 0;
      }
    }
  }

  // Random: randomly defining the epistasis graph
  if (prob_type == 0) {
    v_aux = aloc_vectori(N);

    for (i = 0; i < N; i++) {
      // Initialize vector of output neurons to indices [0, 1, 2, 3, ...]
      for (j = 0; j < N; j++) {
        v_aux[j] = j;
      }

      // For each k, randomly swap output neuron indices
      for (j = 0; j < K; j++) {
        rand_aux = random_int(j, N - 1);
        aux = v_aux[j];
        v_aux[j] = v_aux[rand_aux];
        v_aux[rand_aux] = aux;
      }

      // Set randomly chosen connections to 1 within output matrix
      for (j = 0; j < K; j++) {
        M_out[i][v_aux[j]] = 1;
      }
    }

    delete[] v_aux;
  }
  // Regular: defining the epistasis graph according to the next neighbours
  else if (prob_type == 1) {
    for (i = 0; i < N; i++) {
      L_aux = (i + 1) % N;
      H_aux = (i + K) % N;
      j = L_aux;
      aux = 0;

      // Set neuron and next k connections to 1
      while (aux == 0) {
        M_out[i][j] = 1;
        if (j == H_aux) {
          aux = 1;
        } else {
          j = (j + 1) % N;
        }
      }
    }
  }
  // Semi-regular: randomly defining the epistasis graph according to (an interval of) next neighbours
  else {
    v_aux = aloc_vectori(prob_type + 1);

    for (i = 0; i < N; i++) {
      // Defining the low and high limits of the interval
      L_aux = i - (int)ceil(prob_type / 2.0);

      if (L_aux < 0) {
        L_aux = N + L_aux;
      }

      H_aux = (int)(i + floor(prob_type / 2.0)) % N;

      j = L_aux;
      k = 0;

      // Set aux vector indices within interval
      while (j != ((H_aux + 1) % N)) {
        v_aux[k] = j;
        j = (j + 1) % N;
        k++;
      }

      // For each k, randomly swap output neurons within interval
      for (j = 0; j < K; j++) {
        rand_aux = random_int(j, prob_type);
        aux = v_aux[j];
        v_aux[j] = v_aux[rand_aux];
        v_aux[rand_aux] = aux;
      }

      // Set randomly chosen connections to 1 within output matrix
      for (j = 0; j < K; j++) {
        M_out[i][v_aux[j]] = 1;
      }
    }

    delete[] v_aux;
  }
}

/******************************************************************************\
*				  	Main													 *
\******************************************************************************/
int main(int argc, char *argv[]) {
  int i, j, n_run;
  int N, K;                        // parameter of the NK model: N (size), K (epistasis degree)
  int n_f_cont;                    // n_f_cont = 2^(K+1)
  int prob_type;                   // problem type - 0: random; 1: regular; W>1: semi-regular with window W
  int n_reservoir, reservoir_size; // number of reservoirs, reservoir size
  int n_hid1;                      // size of the hidden layer 1 (n_reservoir * reservoir_size)
  int **M_out;                     // connectivity matrix (defines the connectivity between hidden layer 2 and output layer
  double **F_el;                   // component functions (the number of component functions is N * n_f_cont)

  // Arguments
  if (argc < 6) {
    cout << "Insufficient number of arguments!" << endl;
    cout << "Call: nnet2nk <N> <K> <W> (Prob. type - 0: random; 1: regular; W>1: semi-regular with window W ) <number of reservoirs> <reservoirs size>" << endl;
    exit(1);
  } else {
    N = atoi(argv[1]);
    K = atoi(argv[2]);
    prob_type = atoi(argv[3]);
    n_reservoir = atoi(argv[4]);
    reservoir_size = atoi(argv[5]);

    if (N < 1 || K < 0 || K > N - 1 || n_reservoir < 1 || n_reservoir > 100 || reservoir_size < 1 || reservoir_size > 300 || prob_type < 0 || (prob_type > 1 && prob_type <= K - 1) || prob_type >= N) {
      cout << "Incorrect values for the arguments!" << endl;
      cout << "Call: nnet2nk < N>0 >  < 0<K<N >  < W > (Prob. type - 0: random; 1: regular; W>1: semi-regular with window W ) < 0<number of reservoirs<101 > < 0<reservoirs size<301 >" << endl;
      exit(1);
    }
  }

  // Variables
  n_f_cont = (int)pow(2.0, K + 1.0);
  n_hid1 = reservoir_size * n_reservoir;

  // Allocation for the data to be stored
  F_el = aloc_matrixd(N, n_f_cont);
  M_out = aloc_matrixi(N, N);

  cout << "\n ***** Defining the Neural Networks ****" << endl;
  cout << "N=" << N << ", K=" << K << endl;

  for (n_run = 0; n_run < n_runs_max; n_run++) {
    cout << "Run:" << n_run << "\n"
         << endl;
    srand(n_run + 1);

    // Set connections for probe filter to output layer
    defConn(M_out, prob_type, N, K);

    // Initialize and save network
    nnet *esn = new nnet(n_inputs_inp, n_hid1, N, con_density_inp, spectral_radius_d_inp, reservoir_size, M_out);
    esn->save(n_run);

    // Evaluation of NN i.j
    for (i = 0; i < N; i++) {
      for (j = 0; j < n_f_cont; j++) {
        F_el[i][j] = esn->eval(i, j);
      }
    }

    delete esn;

    // save M_out and F_el
    file_output(F_el, N, K, n_f_cont, M_out, n_run);
  }

  desaloc_matrixi(M_out, N);
  desaloc_matrixd(F_el, N);

  return 0;
}
