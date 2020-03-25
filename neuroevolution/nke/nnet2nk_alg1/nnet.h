/************************************************************************************\
*								 Class: Echo State Network			                 *
* Neural Network with 2 hidden layers; the first one is composed by n_res reservoirs *
\************************************************************************************/
#include "pend.h" // Double Pole class
#include <cmath>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>

#define CHAR_L 1000
#define pi 3.1415926535897932384626433832795

class nnet
{
private:
  double **W_in;  // weight matrix between input and hidden layer 1 (reservoir)
  double **W;     // weight matrix for the recurrent connections in the reservoir (hidden layer 2)
  double **W_hid; // weight matrix between hidden layers 1 and 2
  double **W_out; // weight matrix between hidden layer 2 and output layers
  int n_inp;      // number of units in the input layer
  int n_hid1;     // number of neurons in the hidden layer 1 (reservoir)
  int n_hid2;     // number of neurons in the hidden layer 2
  int n_out;      // number of neurons in the output layer
  typedef struct
  {
    double bias;                                                            // bias
    int n_inputs;                                                           // number of inputs for the neuron
    int *inputs;                                                            // vector with the indices of the input neurons connected to this neuron
    int n_rec;                                                              // number the recurrences in the same layer
    int *rec;                                                               // vector indicating the recurrences in the same layer
  } neuron;                                                                 // candidate partitions used in the partiton crossover
  neuron *hid1_neurons;                                                     // neurons in hidden layer 1 (reservoir)
  neuron *hid2_neurons;                                                     // neurons in hidden layer 2
  neuron *out_neurons;                                                      // neurons in output layer
  void output_nnet(double *u, double *z_old, double *z, double *y, int *x); // compute the outputs of the nnet. u, z_old, z, y, x: respectively, vectors with inputs, old and new activations of the hidden layer 1 units (z_old and z), outputs, solution x controlling the slopes in the hidden layer 2 neurons
public:
  int n_res; // number of reservoirs
  nnet(int n_inp_par, int n_hid_par, int n_out_par, double con_density, double spectral_radius_d, int res_size, int **M_out);
  ~nnet(void);
  double eval(int k, int j); // evaluation of the k-th output of the esn with solution defined by j
  void save(int n_run);
  void print(void);
};

/******************************************************************************\
*								Constructor													   *
\******************************************************************************/
nnet::nnet(int n_inp_par, int n_hid_par, int n_out_par, double con_density, double spectral_radius_d, int res_size, int **M_out)
{
  int i, j, k, k_start, k_end, size_W, k1, k2;
  double spectral_radius, **W_temp, min_W = -0.6, max_W = 0.6;

  n_inp = n_inp_par;
  n_hid1 = n_hid_par;
  n_hid2 = n_out_par; // obs.: hidden layer 2 has the same number of neurons as output layer
  n_out = n_out_par;

  // Memory Allocation
  W_in = aloc_matrixd(n_hid1, n_inp);
  W = aloc_matrixd(n_hid1, n_hid1);
  W_hid = aloc_matrixd(n_hid2, n_hid1);
  W_out = aloc_matrixd(n_out, n_hid2);
  hid1_neurons = new neuron[n_hid1];
  hid2_neurons = new neuron[n_hid2];
  out_neurons = new neuron[n_out];

  // Neurons - Hidden Layer 1
  for (i = 0; i < n_hid1; i++)
  {
    // Inputs
    hid1_neurons[i].n_inputs = n_inp;
    hid1_neurons[i].inputs = aloc_vectori(hid1_neurons[i].n_inputs);
    hid1_neurons[i].bias = (max_W - min_W) * random_dou() + min_W; // random number between min_W and max_W

    for (j = 0; j < n_inp; j++)
    {
      hid1_neurons[i].inputs[j] = j;
      W_in[i][j] = (max_W - min_W) * random_dou() + min_W; // random number between min_W and max_W
    }

    // Initializing Recurrences
    hid1_neurons[i].rec = aloc_vectori(n_hid1);
    hid1_neurons[i].n_rec = 0;

    for (j = 0; j < n_hid1; j++)
    {
      W[i][j] = 0.0;
    }
  }

  // Recurrences
  k_start = 0;
  k_end = res_size - 1;

  if (k_end > n_hid1 - 1)
  {
    k_end = n_hid1 - 1;
  }

  n_res = 1;

  for (i = 0; i < n_hid1; i++)
  {
    if (i > k_end)
    {
      k_start = i;
      k_end = k_start + res_size - 1;

      if (k_end > n_hid1 - 1)
      {
        k_end = n_hid1 - 1;
      }

      n_res++;
    }

    for (j = k_start; j <= k_end; j++)
    {
      if (random_dou() < con_density)
      {
        hid1_neurons[i].rec[hid1_neurons[i].n_rec] = j;
        hid1_neurons[i].n_rec = hid1_neurons[i].n_rec + 1;
        W[i][j] = (max_W - min_W) * random_dou() + min_W; // random number between min_W and max_W
      }
    }
  }

  // Scaling W to spectral_radius_d W
  k_start = 0;
  k_end = res_size - 1;

  if (k_end > n_hid1 - 1)
  {
    k_end = n_hid1 - 1;
  }

  for (k = 0; k < n_res; k++)
  {
    // W_temp: W for reservoir k
    size_W = k_end - k_start + 1;
    W_temp = aloc_matrixd(size_W, size_W);
    i = 0;
    for (k1 = k_start; k1 <= k_end; k1++)
    {
      j = 0;
      for (k2 = k_start; k2 <= k_end; k2++)
      {
        W_temp[i][j] = W[k1][k2];
        j++;
      }
      i++;
    }

    // Computing the spectral radius of W_temp
    spectral_radius = largEig(W_temp, size_W, size_W);

    // Normalizing W to desired spectral radius (Scaling W to spectral_radius_d (1/spectral_radius) W)
    for (i = 0; i < size_W; i++)
    {
      for (j = 0; j < size_W; j++)
      {
        W_temp[i][j] = spectral_radius_d * W_temp[i][j] / spectral_radius;
      }
    }

    i = 0;
    for (k1 = k_start; k1 <= k_end; k1++)
    {
      j = 0;
      for (k2 = k_start; k2 <= k_end; k2++)
      {
        W[k1][k2] = W_temp[i][j];
        j++;
      }
      i++;
    }

    k_start = k_end + 1;
    k_end = k_start + res_size - 1;

    if (k_end > n_hid1 - 1)
    {
      k_end = n_hid1 - 1;
    }

    desaloc_matrixd(W_temp, size_W);
  }

  // Neurons - Hidden Layer 2
  for (i = 0; i < n_hid2; i++)
  {
    hid2_neurons[i].n_inputs = 0;
    hid2_neurons[i].inputs = aloc_vectori(n_hid1);
    hid2_neurons[i].n_rec = 0;
    hid2_neurons[i].bias = (max_W - min_W) * random_dou() + min_W; // random number between min_W and max_W

    for (j = 0; j < n_hid1; j++)
    {
      hid2_neurons[i].inputs[hid2_neurons[i].n_inputs] = j;
      hid2_neurons[i].n_inputs = hid2_neurons[i].n_inputs + 1;
      W_hid[i][j] = (max_W - min_W) * random_dou() + min_W; // random number between min_W and max_W
    }
  }

  // Neurons - Output Layer (remember that the bias of each output neuron is given by the nk model
  for (i = 0; i < n_out; i++)
  {
    for (j = 0; j < n_hid2; j++)
    {
      W_out[i][j] = 0.0;
    }
  }

  for (i = 0; i < n_out; i++)
  {
    out_neurons[i].n_inputs = 0;
    out_neurons[i].inputs = aloc_vectori(n_hid2);
    out_neurons[i].n_rec = 0;
    out_neurons[i].bias = (max_W - min_W) * random_dou() + min_W; // random number between min_W and max_W

    for (j = 0; j < n_hid2; j++)
    {
      if (M_out[i][j] == 1)
      {
        out_neurons[i].inputs[out_neurons[i].n_inputs] = j;
        out_neurons[i].n_inputs = out_neurons[i].n_inputs + 1;
        W_out[i][j] = (max_W - min_W) * random_dou() + min_W; // random number between min_W and max_W
      }
    }
  }
}

/******************************************************************************\
*								 Destructor													   *
\******************************************************************************/
nnet::~nnet(void)
{
  int i;

  desaloc_matrixd(W_in, n_hid1);
  desaloc_matrixd(W, n_hid1);
  desaloc_matrixd(W_hid, n_hid2);
  desaloc_matrixd(W_out, n_out);

  for (i = 0; i < n_hid1; i++)
  {
    delete[] hid1_neurons[i].inputs;
    delete[] hid1_neurons[i].rec;
  }

  delete[] hid1_neurons;

  for (i = 0; i < n_hid2; i++)
  {
    delete[] hid2_neurons[i].inputs;
  }

  delete[] hid2_neurons;

  for (i = 0; i < n_out; i++)
  {
    delete[] out_neurons[i].inputs;
  }

  delete[] out_neurons;
}

/******************************************************************************\
*					 Output of the Neural netwok (y) for input u and sol. x			   *
\******************************************************************************/
void nnet::output_nnet(double *u, double *z_old, double *z, double *y, int *x)
{
  int i, j;
  double *s, sum_u;

  s = aloc_vectord(n_hid2); // outputs of the hidden layer 2

  // Activation of the neurons in the hidden layer 1 (reservoir)
  for (i = 0; i < n_hid1; i++)
  {
    sum_u = hid1_neurons[i].bias;

    for (j = 0; j < n_inp; j++)
    {
      sum_u = sum_u + u[j] * W_in[i][j];
    }

    for (j = 0; j < hid1_neurons[i].n_rec; j++)
    {
      sum_u = sum_u + z_old[hid1_neurons[i].rec[j]] * W[i][hid1_neurons[i].rec[j]];
    }

    z[i] = tanh(1.0 * sum_u); // Tangent hiperbolic with half-slope a=2
  }

  // Activation of the neurons in the hidden layer 2
  for (i = 0; i < n_hid2; i++)
  {
    sum_u = hid2_neurons[i].bias;

    for (j = 0; j < n_hid1; j++)
    {
      sum_u = sum_u + z[hid2_neurons[i].inputs[j]] * W_hid[i][hid2_neurons[i].inputs[j]];
    }

    if (x[i] == 0)
    {
      s[i] = 0.0 * tanh(1.0 * sum_u); // Tangent hiperbolic with half-slope 1 and input sum_u; obs.: in the original tanh, a=1
    }
    else
    {
      s[i] = tanh(1.0 * sum_u); // Tangent hiperbolic with half-slope 3 and input sum_u; obs.: in the original tanh, a=3
    }
  }

  // Activation of the output units
  for (i = 0; i < n_out; i++)
  {
    sum_u = out_neurons[i].bias;

    for (j = 0; j < out_neurons[i].n_inputs; j++)
    {
      sum_u = sum_u + s[out_neurons[i].inputs[j]] * W_out[i][out_neurons[i].inputs[j]];
    }

    y[i] = tanh(1.0 * sum_u); // Tangent hyperbolic with half-slope 2 and input sum_u; obs.: in the original tanh, a=2
  }

  delete[] s;
}

/******************************************************************************\
*	Evaluation of the k-th output of the esn with sol. defined by j		        *
\******************************************************************************/
double nnet::eval(int k, int j)
{
  int i, t, *sol, *sol_reduced;
  double st[6], max_teta, max_x, fit = 0.0, *z, *z_init, *z_old, *y, u[3], F, fit_stable, sum_fst, f_cheap, *fst;
  doublePole *Pole; // declaring Inverted Double Pole class variable

  // Memory allocation
  z = aloc_vectord(n_hid1);
  z_init = aloc_vectord(n_hid1);
  z_old = aloc_vectord(n_hid1);
  y = aloc_vectord(n_out);
  fst = aloc_vectord(t_max);
  sol = aloc_vectori(n_hid2);
  sol_reduced = aloc_vectori(out_neurons[k].n_inputs);

  // Maximum values
  max_teta = 36 * (2 * pi) / 360; // (rad) maximum value for the pole angles
  max_x = 2.4;                    // (m) maximum value for the cart position (track size)

  // Solution
  dec2binvec(j, sol_reduced, out_neurons[k].n_inputs);

  for (i = 0; i < n_hid2; i++)
  {
    sol[i] = 0;
  }

  for (i = 0; i < out_neurons[k].n_inputs; i++)
  {
    sol[out_neurons[k].inputs[i]] = sol_reduced[i];
  }

  // Initial inputs for the NNet
  for (i = 0; i < n_inp; i++)
  {
    u[i] = 0.0;
  }

  // Initial hidden neuron activations
  for (i = 0; i < n_hid1; i++)
  {
    z_init[i] = 0.0;
  }

  // Running the NNet for 20 iterations with zero input
  for (t = 0; t < 20; t++)
  {
    output_nnet(u, z_init, z, y, sol);

    for (i = 0; i < n_hid1; i++)
    {
      z_init[i] = z[i];
    }
  }

  // Initialization of the evaluation
  // States (st)
  st[0] = 0.0;                  // (m) cart position: x
  st[1] = 0.0;                  // (m/s) cart velocity: dx
  st[2] = 4.5 * (2 * pi) / 360; // (rad) angle - pole 1
  st[3] = 0.0;                  // (rad/s) ang. velocity - pole 1
  st[4] = 0.0;                  // (rad) angle - pole 2
  st[5] = 0.0;                  // (rad/s) ang. velocity - pole 2
  t = 0;                        // iteration

  for (i = 0; i < n_hid1; i++)
  {
    z_old[i] = z_init[i];
  }

  Pole = new doublePole(st); // initialization Inverted Double Pole
  sum_fst = 0.0;

  // Simulation of the Double Pole
  while (t < t_max && fabs(st[0]) < max_x && fabs(st[2]) < max_teta && fabs(st[4]) < max_teta)
  {
    u[0] = st[0] / max_x;    // cart position: x
    u[1] = st[2] / max_teta; // angle - pole 1
    u[2] = st[4] / max_teta; // angle - pole 2

    output_nnet(u, z_old, z, y, sol);

    for (i = 0; i < n_hid1; i++)
    {
      z_old[i] = z[i];
    }

    F = y[k] * 10.0;   // scaling F in the range [-10,10]
    Pole->step(F, st); // step of the inverted double pole dynamical system

    // compute f_cheap
    fst[t] = fabs(st[0]) + fabs(st[1]) + fabs(st[2]) + fabs(st[3]);

    if (t <= 100)
    {
      fit_stable = 0.0;
      sum_fst = sum_fst + fst[t];
    }
    else
    {
      sum_fst = sum_fst + fst[t] - fst[t - 101];
      fit_stable = 0.75 / sum_fst;
    }

    f_cheap = 0.0001 * t + 0.9 * fit_stable;

    Pole->print();

    t++;
  }

  //	cout<<"t: "<<t<<endl;

  fit = f_cheap;

  delete Pole;
  delete[] y;
  delete[] z;
  delete[] z_init;
  delete[] z_old;
  delete[] fst;
  delete[] sol;
  delete[] sol_reduced;

  return (fit);
}

/******************************************************************************\
*								Save NNet information														   *
\******************************************************************************/
void nnet::save(int n_run)
{
  int i, j;
  FILE *NN_file;
  char *name_p;
  char name[CHAR_L];

  name_p = name;

  sprintf(name, "data/NN_%d.dat", n_run);

  if ((NN_file = fopen(name_p, "w")) == NULL)
  {
    puts("The file NN to be saved cannot be open \n");
    exit(1);
  }

  // Save NN information
  fprintf(NN_file, "Parameters\n");
  fprintf(NN_file, "%d %d %d %d \n", n_inp, n_hid1, n_hid2, n_out);

  // Save W_in
  fprintf(NN_file, "W_in\n");
  for (i = 0; i < n_hid1; i++)
  {
    for (j = 0; j < n_inp; j++)
    {
      fprintf(NN_file, "%1.14f ", W_in[i][j]);
    }
    fprintf(NN_file, "\n");
  }

  // Save W
  fprintf(NN_file, "W_rec\n");
  for (i = 0; i < n_hid1; i++)
  {
    for (j = 0; j < n_hid1; j++)
    {
      fprintf(NN_file, "%1.14f ", W[i][j]);
    }
    fprintf(NN_file, "\n");
  }

  // Save W_hid
  fprintf(NN_file, "W_hid\n");
  for (i = 0; i < n_hid2; i++)
  {
    for (j = 0; j < n_hid1; j++)
    {
      fprintf(NN_file, "%1.14f ", W_hid[i][j]);
    }
    fprintf(NN_file, "\n");
  }

  // Save W_out
  fprintf(NN_file, "W_out\n");
  for (i = 0; i < n_out; i++)
  {
    for (j = 0; j < n_hid2; j++)
    {
      fprintf(NN_file, "%1.14f ", W_out[i][j]);
    }
    fprintf(NN_file, "\n");
  }

  // Save bias neurons layer hid1
  fprintf(NN_file, "B_hid1\n");
  for (i = 0; i < n_hid1; i++)
  {
    fprintf(NN_file, "%1.14f ", hid1_neurons[i].bias);
  }
  fprintf(NN_file, "\n");

  // Save bias neurons layer hid2
  fprintf(NN_file, "B_hid2\n");
  for (i = 0; i < n_hid2; i++)
  {
    fprintf(NN_file, "%1.14f ", hid2_neurons[i].bias);
  }
  fprintf(NN_file, "\n");

  // Save bias neurons layer out
  fprintf(NN_file, "B_out\n");
  for (i = 0; i < n_out; i++)
  {
    fprintf(NN_file, "%1.14f ", out_neurons[i].bias);
  }
  fprintf(NN_file, "\n");

  fclose(NN_file);
}

/******************************************************************************\
*								Save NNet information														   *
\******************************************************************************/
/*void nnet::save(int n_run){
	int i, j;
	FILE *Win_file, *Wre_file, *Whi_file, *Wou_file, *Bh1_file, *Bh2_file, *Bou_file, *NNi_file;
	char *name_p;
	char name[CHAR_L];

    name_p = name;

    // Save W_in
	sprintf(name,"NN_data/Win_%d.dat",n_run);
	if ((Win_file = fopen(name_p,"w"))==NULL) {
		puts("The file Win to be saved cannot be open \n");
		exit(1);
	}
	for (i=0;i<n_hid1;i++){
	      for (j=0;j<n_inp;j++)
		        fprintf(Win_file,"%f ",W_in[i][j]);
          fprintf(Win_file,"\n");         
	}
	fclose(Win_file);
	
	 // Save W
	sprintf(name,"NN_data/Wre_%d.dat",n_run);
	if ((Wre_file = fopen(name_p,"w"))==NULL) {
		puts("The file Wre to be saved cannot be open \n");
		exit(1);
	}
	for (i=0;i<n_hid1;i++){
	      for (j=0;j<n_hid1;j++)
		        fprintf(Wre_file,"%f ",W[i][j]);
          fprintf(Wre_file,"\n");         
	}
	fclose(Wre_file);
	
	// Save W_hid
	sprintf(name,"NN_data/Whi_%d.dat",n_run);
	if ((Whi_file = fopen(name_p,"w"))==NULL) {
		puts("The file Whi to be saved cannot be open \n");
		exit(1);
	}
	for (i=0;i<n_hid2;i++){
	      for (j=0;j<n_hid1;j++)
		        fprintf(Whi_file,"%f ",W_hid[i][j]);
          fprintf(Whi_file,"\n");         
	}
	fclose(Whi_file);
	
	// Save W_out
	sprintf(name,"NN_data/Wou_%d.dat",n_run);
	if ((Wou_file = fopen(name_p,"w"))==NULL) {
		puts("The file Wou to be saved cannot be open \n");
		exit(1);
	}
	for (i=0;i<n_out;i++){
	      for (j=0;j<n_hid2;j++)
		        fprintf(Wou_file,"%f ",W_out[i][j]);
          fprintf(Wou_file,"\n");         
	}
	fclose(Wou_file);
	
	// Save bias neurons layer hid1
	sprintf(name,"NN_data/Bh1_%d.dat",n_run);
	if ((Bh1_file = fopen(name_p,"w"))==NULL) {
		puts("The file Bh1 to be saved cannot be open \n");
		exit(1);
	}
	for (i=0;i<n_hid1;i++)
        fprintf(Bh1_file,"%f ",hid1_neurons[i].bias);
    fprintf(Bh1_file,"\n");         
	fclose(Bh1_file);
	
	// Save bias neurons layer hid2
	sprintf(name,"NN_data/Bh2_%d.dat",n_run);
	if ((Bh2_file = fopen(name_p,"w"))==NULL) {
		puts("The file Bh2 to be saved cannot be open \n");
		exit(1);
	}
	for (i=0;i<n_hid2;i++)
        fprintf(Bh2_file,"%f ",hid2_neurons[i].bias);
    fprintf(Bh2_file,"\n");         
	fclose(Bh2_file);
	
	// Save bias neurons layer out
	sprintf(name,"NN_data/Bou_%d.dat",n_run);
	if ((Bou_file = fopen(name_p,"w"))==NULL) {
		puts("The file Bou to be saved cannot be open \n");
		exit(1);
	}
	for (i=0;i<n_out;i++)
        fprintf(Bou_file,"%f ",out_neurons[i].bias);
    fprintf(Bou_file,"\n");         
	fclose(Bou_file);
	
	// Save NN information
	sprintf(name,"NN_data/NNi_%d.dat",n_run);
	if ((NNi_file = fopen(name_p,"w"))==NULL) {
		puts("The file NNi to be saved cannot be open \n");
		exit(1);
	}
	fprintf(NNi_file,"%d %d %d %d \n",n_inp, n_hid1, n_hid2, n_out);        
	fclose(NNi_file);	
	
}*/

/******************************************************************************\
*								Print NNet information														   *
\******************************************************************************/
void nnet::print(void)
{
  int i, j;

  cout << "Echo State Network: " << endl;
  cout << " Number of Inputs: " << n_inp << endl;
  cout << " Number of Neurons in Hidden Layer 1 (reservoir): " << n_hid1 << endl;
  cout << " Number of Neurons in Hidden Layer 2: " << n_hid2 << endl;
  cout << " Number of Outputs: " << n_out << endl;
  cout << " Hidden Layer 1: " << endl;

  for (i = 0; i < n_hid1; i++)
  {
    cout << " Neuron: " << i << " , bias: " << hid1_neurons[i].bias << endl;
    cout << "  W_in: ";
    for (j = 0; j < n_inp; j++)
    {
      cout << W_in[i][j] << ", ";
    }
    cout << endl;
    cout << "  W: ";
    for (j = 0; j < n_hid1; j++)
    {
      cout << W[i][j] << ", ";
    }
    cout << endl;
  }

  cout << " Hidden Layer 2: " << endl;

  for (i = 0; i < n_hid2; i++)
  {
    cout << " Neuron: " << i << " , bias: " << hid2_neurons[i].bias << endl;
    cout << "  W_hid: ";
    for (j = 0; j < n_hid1; j++)
    {
      cout << W_hid[i][j] << ", ";
    }
    cout << endl;
  }

  cout << " Output Layer: " << endl;

  for (i = 0; i < n_out; i++)
  {
    cout << " Neuron: " << i << " , bias: " << out_neurons[i].bias << endl;
    cout << "  W_out: ";
    for (j = 0; j < n_hid2; j++)
    {
      cout << W_out[i][j] << ", ";
    }
    cout << endl;
  }
}
