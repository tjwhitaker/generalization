/************************************************************************************\
*								 Class: Echo State Network			                 *
* Neural Network with 2 hidden layers; the first one is composed by n_res reservoirs *
\************************************************************************************/
#include <cmath>
#include <cstdlib>
#include "pend.h"			// Double Pole class
#include <stdio.h>
#include <stdlib.h>
#include<string>
#include<cstring>
#include<fstream>

#define CHAR_LEN 1000
#define pi 3.1415926535897932384626433832795 

class nnetCom {
private:	
	double **W_in;										// weight matrix between input and hidden layer 1 (reservoir)
	double **W;											// weight matrix for the recurrent connections in the reservoir (hidden layer 2)
	double **W_hid;										// weight matrix between hidden layers 1 and 2
	double **W_out;										// weight matrix between hidden layer 2 and output layers
	double *w_com;                                      // weight vector for the outputs of the committee
	int n_inp;											// number of units in the input layer
	int n_hid1;											// number of neurons in the hidden layer 1 (reservoir)
	int n_out;										    // number of neurons in the output layer
	typedef struct{	
		double bias;			// bias
		int n_inputs;			// number of inputs for the neuron
		int *inputs;			// vector with the indices of the input neurons connected to this neuron
		int n_rec;				// number the recurrences in the same layer
		int *rec;				// vector indicating the recurrences in the same layer
	} neuron;			        // candidate partitions used in the partiton crossover
	neuron *hid1_neurons;       // neurons in hidden layer 1 (reservoir)
	neuron *hid2_neurons;       // neurons in hidden layer 2
	neuron *out_neurons;        // neurons in output layer
	void output_nnet(double *u, double *z_old, double *z, double *y, int *x);		// compute the outputs of the nnet. u, z_old, z, y, x: respectively, vectors with inputs, old and new activations of the hidden layer 1 units (z_old and z), outputs, solution x controlling the slopes in the hidden layer 2 neurons  
public:
	int n_hid2;											// number of neurons in the hidden layer 2
	int *res_test;                                      // vector with the results of the test
	nnetCom(int n_run);
    ~nnetCom(void);
	double eval(int *sol, int i_out);		// evaluation of the NN with solution sol
	double evalGen(int *sol, int i_out, int com_esem);	// evaluation of the NN with solution sol: generalization test
	void findWeights(int *sol);
	void findWeights2(int *sol);
    void print(void);
};

/******************************************************************************\
*								Constructor													   *
\******************************************************************************/
nnetCom::nnetCom(int n_run){
	int i, j;
	char line[CHAR_LEN], * keywords,Delimiters[] = " :=\n\t\r\f\v", name[CHAR_LEN], *name_p;
	
	res_test=aloc_vectori(625);
	
    // Read file
    name_p = name;
    sprintf(name,"data/NN_%d.dat",n_run);

    ifstream fin(name_p);
    while((fin.getline(line, CHAR_LEN-1))){
		if(!(keywords = strtok(line, Delimiters)))
  			continue;
		if(!strcmp(keywords, "Parameters")){
            fin>>n_inp;
            fin>>n_hid1;
            fin>>n_hid2;
            fin>>n_out;
            W_in=aloc_matrixd (n_hid1,n_inp);		
	        W=aloc_matrixd (n_hid1,n_hid1);
	        W_hid=aloc_matrixd (n_hid2,n_hid1);
	        W_out=aloc_matrixd (n_out,n_hid2);		
	        hid1_neurons= new neuron [n_hid1]; 
            hid2_neurons= new neuron [n_hid2];	
	        out_neurons= new neuron [n_out];  
            w_com=aloc_vectord(n_out);			
		} 
		else if (!strcmp(keywords, "W_in")){
              for (i=0;i<n_hid1;i++)
                for (j=0;j<n_inp;j++)                        
		           fin>>W_in[i][j];                			 
        }
        else if (!strcmp(keywords, "W_rec")){
             for (i=0;i<n_hid1;i++)
                for (j=0;j<n_hid1;j++)                        
		           fin>>W[i][j];              
        }
         else if (!strcmp(keywords, "W_hid")){
             for (i=0;i<n_hid2;i++)
                for (j=0;j<n_hid1;j++)                        
		           fin>>W_hid[i][j];              
        }
        else if (!strcmp(keywords, "W_out")){
             for (i=0;i<n_out;i++)
                for (j=0;j<n_hid2;j++)                        
		           fin>>W_out[i][j];              
        }
         else if (!strcmp(keywords, "B_hid1")){
             for (i=0;i<n_hid1;i++)                    
		           fin>>hid1_neurons[i].bias;              
        }
        else if (!strcmp(keywords, "B_hid2")){
             for (i=0;i<n_hid2;i++)                    
		           fin>>hid2_neurons[i].bias;              
        }
        else if (!strcmp(keywords, "B_out")){
             for (i=0;i<n_out;i++)                    
		           fin>>out_neurons[i].bias;              
        }
  }
 
  fin.close();
			
}


/******************************************************************************\
*								 Destructor													   *
\******************************************************************************/
nnetCom::~nnetCom(void){	
	
	desaloc_matrixd (W_in, n_hid1);
	desaloc_matrixd (W,n_hid1);
	desaloc_matrixd (W_hid,n_hid2);
	desaloc_matrixd (W_out,n_out);
	delete [] hid1_neurons;
	delete [] hid2_neurons;
	delete [] out_neurons;
	delete [] w_com;
	delete [] res_test;

}  


/******************************************************************************\
*					 Output of the Neural netwok (y) for input u and sol. x			   *
\******************************************************************************/
void nnetCom::output_nnet(double *u, double *z_old, double *z, double *y, int *x){
		int i, j;
		double *s, sum_u;
	
        s=aloc_vectord(n_hid2);                 // outputs of the hidden layer 2
        
		// Activation of the neurons in the hidden layer 1 (reservoir)
		for (i=0;i<n_hid1;i++){
			sum_u=hid1_neurons[i].bias ;
			for (j=0;j<n_inp;j++)
				sum_u = sum_u + u[j]*W_in[i][j];
			for (j=0 ; j<n_hid1; j++)
				sum_u = sum_u + z_old[ j ]*W[i][j];			
			z[i]=tanh(1.0*sum_u);				// Tangent hiperbolic with half-slope a=2	
         
		}
	
		// Activation of the neurons in the hidden layer 2
		for (i=0;i<n_hid2;i++){
			sum_u=hid2_neurons[i].bias ;
			for (j=0;j<n_hid1;j++)
				sum_u = sum_u + z[ j ]*W_hid[i][ j ];		
			if (x[i]==0)
				s[i]=0.0*tanh(1.0*sum_u);				// Tangent hiperbolic with half-slope 1 and input sum_u; obs.: in the original tanh, a=1	
			else
				s[i]=tanh(1.0*sum_u);				// Tangent hiperbolic with half-slope 3 and input sum_u; obs.: in the original tanh, a=3	
		
		}
		// Activation of the output units
		for (i=0;i<n_out;i++){
			sum_u=out_neurons[i].bias ;
			for (j=0; j<n_hid2 ;j++)
				sum_u = sum_u + s[ j ]*W_out[i][ j ];	
			y[i]=tanh(1.0*sum_u);				// Tangent hiperbolic with half-slope 2 and input sum_u; obs.: in the original tanh, a=2	
		}	
        
        delete [] s;		
	
}

/******************************************************************************\
*	Evaluation of esn with sol. defined by j		        *
\******************************************************************************/
void nnetCom::findWeights2(int *sol){
  	int i, t,j;
	double st[6], max_teta, max_x, sum_fit=0.0, *z, *z_init, *z_old, *y, u[3], F, fit_stable, sum_fst, f_cheap, *fst, *fitness;
	doublePole *Pole;		// declaring Inverted Double Pole class variable
	
	
	// Memory allocation
	z=aloc_vectord (n_hid1);
	z_init=aloc_vectord (n_hid1);
	z_old=aloc_vectord (n_hid1);
	y=aloc_vectord (n_out);
	fst=aloc_vectord(t_max);
	fitness=aloc_vectord(n_out);

	
	// Maximum values
	max_teta=36*(2*pi)/360;		    // (rad) maximum value for the pole angles
	max_x=2.4;					// (m) maximum value for the cart position (track size)
  
	// Initial inputs for the NNet
	for (i=0;i<n_inp;i++)
		u[i]=0.0;
	// Initial hidden neuron activations
	for (i=0;i<n_hid1;i++)
		z_init[i]=0.0;
	
	// Running the NNet for 20 iterations with zero input
	for (t=0;t<20;t++){
		output_nnet(u,z_init,z,y,sol);
		for (i=0;i<n_hid1;i++)
			z_init[i]=z[i];		
	}
	
	for (j=0;j<n_out;j++){
        // Initialization of the evaluation
        // States (st)
        st[0]=0.0;					// (m) cart position: x
        st[1]=0.0;					// (m/s) cart velocity: dx
	    st[2]=4.5*(2*pi)/360;	        // (rad) angle - pole 1
	    st[3]=0.0;                  // (rad/s) ang. velocity - pole 1
	    st[4]=0.0;	                // (rad) angle - pole 2
	    st[5]=0.0;                  // (rad/s) ang. velocity - pole 2
    	t=0;						// iteration
    	for (i=0;i<n_hid1;i++)
    		z_old[i]=z_init[i];	
           
    
    	Pole = new doublePole(st);	// initialization Inverted Double Pole
    	sum_fst=0.0;
    		
    	// Simulation of the Double Pole
    	while(t<t_max && fabs(st[0])<max_x && fabs(st[2])<max_teta && fabs(st[4])<max_teta  ){
    			u[0]=st[0]/max_x;       // cart position: x                 
			    u[1]=st[2]/max_teta;       // angle - pole 1
			    u[2]=st[4]/max_teta;       // angle - pole 2
    			output_nnet(u,z_old,z,y,sol);
    			for (i=0;i<n_hid1;i++)
    	            z_old[i]=z[i];	
    	           
                // comput output
                F=y[j]*10.0;	                 
    		
    			Pole->step(F,st);				  // step of the inverted double pole dynamical system
          		// compute f_cheap
    			fst[t]=fabs(st[0])+fabs(st[1])+fabs(st[2])+fabs(st[3]);
    			if (t<=100){
    				fit_stable=0.0;
    				sum_fst=sum_fst+fst[t];
    			}
    			else{
    				sum_fst=sum_fst+fst[t]-fst[t-101];
    				fit_stable=0.75/sum_fst;
    			}
    			f_cheap=0.0001*t +0.9*fit_stable;
    			//Pole->print();		
    	    	t++;
    	}
        fitness[j]=f_cheap;
      //  cout<<endl<<"fit["<<j<<"]: "<<fitness[j];
        //cout<<t<<", ";
        //sum_fit=sum_fit+fitness[j];
        
        delete Pole;
    }
  //  cout<<endl<<"mean: "<<sum_fit/n_out<<endl;

    /*for (i=0;i<n_out;i++)
        w_com[i]=fitness[i]/sum_fit;  */
          
   int *v_aux;
   v_aux = aloc_vectori(n_out);
   for (i=0;i<n_out;i++)
       v_aux[i]=i;
   int size_s=n_out/5;
   int x_aux, i_max;
   for (i=0;i<size_s;i++){
       i_max=i;
       for (j=i+1;j<n_out;j++){
           if (fitness[v_aux[j]]>fitness[v_aux[i_max]])
            i_max=j;
       }
       x_aux=v_aux[i];
       v_aux[i]=v_aux[i_max];       
       v_aux[i_max] = x_aux;
   }
   sum_fit=0.0;
   for (i=0;i<size_s;i++)
       sum_fit=sum_fit+fitness[v_aux[i]];
   for (i=0;i<size_s;i++)
       w_com[v_aux[i]]=fitness[v_aux[i]]/sum_fit;       
   for (i=size_s;i<n_out;i++)
        w_com[v_aux[i]]=0.0;
   
  // for (i=0;i<n_out;i++)
   //    cout<<fitness[i]<<"->"<<w_com[i]<<endl;
   
   delete [] v_aux;
	
	delete [] y;
	delete [] z;
	delete [] z_init;
	delete [] z_old;
	delete [] fst;
    delete [] fitness;	
}

/******************************************************************************\
*	Evaluation of esn with sol. defined by j		        *
\******************************************************************************/
void nnetCom::findWeights(int *sol){
  	int i, t,j;
	double st[6], max_teta, max_x, sum_fit=0.0, *z, *z_init, *z_old, *y, u[3], F, fit_stable, sum_fst, f_cheap, *fst, *fitness;
	doublePole *Pole;		// declaring Inverted Double Pole class variable
	
	
	// Memory allocation
	z=aloc_vectord (n_hid1);
	z_init=aloc_vectord (n_hid1);
	z_old=aloc_vectord (n_hid1);
	y=aloc_vectord (n_out);
	fst=aloc_vectord(t_max);
	fitness=aloc_vectord(n_out);

	
	// Maximum values
	max_teta=36*(2*pi)/360;		    // (rad) maximum value for the pole angles
	max_x=2.4;					// (m) maximum value for the cart position (track size)
  
	// Initial inputs for the NNet
	for (i=0;i<n_inp;i++)
		u[i]=0.0;
	// Initial hidden neuron activations
	for (i=0;i<n_hid1;i++)
		z_init[i]=0.0;
	
	// Running the NNet for 20 iterations with zero input
	for (t=0;t<20;t++){
		output_nnet(u,z_init,z,y,sol);
		for (i=0;i<n_hid1;i++)
			z_init[i]=z[i];		
	}
	
	for (j=0;j<n_out;j++){
        // Initialization of the evaluation
        // States (st)
        st[0]=0.0;					// (m) cart position: x
        st[1]=0.0;					// (m/s) cart velocity: dx
	    st[2]=4.5*(2*pi)/360;	        // (rad) angle - pole 1
	    st[3]=0.0;                  // (rad/s) ang. velocity - pole 1
	    st[4]=0.0;	                // (rad) angle - pole 2
	    st[5]=0.0;                  // (rad/s) ang. velocity - pole 2
    	t=0;						// iteration
    	for (i=0;i<n_hid1;i++)
    		z_old[i]=z_init[i];	
           
    
    	Pole = new doublePole(st);	// initialization Inverted Double Pole
    	sum_fst=0.0;
    		
    	// Simulation of the Double Pole
    	while(t<t_max && fabs(st[0])<max_x && fabs(st[2])<max_teta && fabs(st[4])<max_teta  ){
    			u[0]=st[0]/max_x;       // cart position: x                 
			    u[1]=st[2]/max_teta;       // angle - pole 1
			    u[2]=st[4]/max_teta;       // angle - pole 2
    			output_nnet(u,z_old,z,y,sol);
    			for (i=0;i<n_hid1;i++)
    	            z_old[i]=z[i];	
    	           
                // comput output
                F=y[j]*10.0;	                 
    		
    			Pole->step(F,st);				  // step of the inverted double pole dynamical system
          		// compute f_cheap
    			fst[t]=fabs(st[0])+fabs(st[1])+fabs(st[2])+fabs(st[3]);
    			if (t<=100){
    				fit_stable=0.0;
    				sum_fst=sum_fst+fst[t];
    			}
    			else{
    				sum_fst=sum_fst+fst[t]-fst[t-101];
    				fit_stable=0.75/sum_fst;
    			}
    			f_cheap=0.0001*t +0.9*fit_stable;
    			//Pole->print();		
    	    	t++;
    	}
        fitness[j]=f_cheap;
      //  cout<<endl<<"fit["<<j<<"]: "<<fitness[j];
        //cout<<t<<", ";
        sum_fit=sum_fit+fitness[j];
        
        delete Pole;
    }
  //  cout<<endl<<"mean: "<<sum_fit/n_out<<endl;

    for (i=0;i<n_out;i++)
        w_com[i]=fitness[i]/sum_fit;    
   /*int *v_aux;
   v_aux = aloc_vectori(n_out);
   for (i=0;i<n_out;i++)
       v_aux[i]=i;
   int size_s=n_out/2;
   int x_aux, i_max;
   for (i=0;i<size_s;i++){
       i_max=i;
       for (j=i+1;j<n_out;j++){
           if (fitness[v_aux[j]]>fitness[v_aux[i_max]])
            i_max=j;
       }
       x_aux=v_aux[i];
       v_aux[i]=v_aux[i_max];       
       v_aux[i_max] = x_aux;
   }
   sum_fit=0.0;
   for (i=0;i<size_s;i++)
       sum_fit=sum_fit+fitness[v_aux[i]];
   for (i=0;i<size_s;i++)
       w_com[v_aux[i]]=fitness[v_aux[i]]/sum_fit;       
   for (i=size_s;i<n_out;i++)
        w_com[v_aux[i]]=0.0;
   
  // for (i=0;i<n_out;i++)
   //    cout<<fitness[i]<<"->"<<w_com[i]<<endl;
   
   delete [] v_aux;*/
	
	delete [] y;
	delete [] z;
	delete [] z_init;
	delete [] z_old;
	delete [] fst;
    delete [] fitness;	
}

/******************************************************************************\
*	Evaluation of esn with sol. defined by j		        *
\******************************************************************************/
/*void nnetCom::findWeights(int *sol){
     int i; 
     
     for (i=0;i<n_out;i++)
       w_com[i]=1.0/n_out;     


}*/

/******************************************************************************\
*	Evaluation of esn with sol. defined by j		        *
\******************************************************************************/
double nnetCom::eval(int *sol, int i_out){
  	int i, t, vote;
	double st[6], max_teta, max_x, fit=0.0, *z, *z_init, *z_old, *y, u[3], F, fit_stable, sum_fst, f_cheap, *fst, w_com_sum, vote_sum;
	doublePole *Pole;		// declaring Inverted Double Pole class variable
	
	
	// Memory allocation
	z=aloc_vectord (n_hid1);
	z_init=aloc_vectord (n_hid1);
	z_old=aloc_vectord (n_hid1);
	y=aloc_vectord (n_out);
	fst=aloc_vectord(t_max);

	
	// Maximum values
	max_teta=36*(2*pi)/360;		    // (rad) maximum value for the pole angles
	max_x=2.4;					// (m) maximum value for the cart position (track size)
    
	// Initial inputs for the NNet
	for (i=0;i<n_inp;i++)
		u[i]=0.0;
	// Initial hidden neuron activations
	for (i=0;i<n_hid1;i++)
		z_init[i]=0.0;
	
	// Running the NNet for 20 iterations with zero input
	for (t=0;t<20;t++){
		output_nnet(u,z_init,z,y,sol);
		for (i=0;i<n_hid1;i++)
			z_init[i]=z[i];		
	}
	
    // Initialization of the evaluation
    // States (st)
    st[0]=0.0;					// (m) cart position: x
    st[1]=0.0;					// (m/s) cart velocity: dx
	st[2]=4.5*(2*pi)/360;	        // (rad) angle - pole 1
	st[3]=0.0;                  // (rad/s) ang. velocity - pole 1
	st[4]=0.0;	                // (rad) angle - pole 2
	st[5]=0.0;                  // (rad/s) ang. velocity - pole 2
	t=0;						// iteration
	for (i=0;i<n_hid1;i++)
		z_old[i]=z_init[i];	
       

	Pole = new doublePole(st);	// initialization Inverted Double Pole
	sum_fst=0.0;
		
	// Simulation of the Double Pole
	while(t<t_max && fabs(st[0])<max_x && fabs(st[2])<max_teta && fabs(st[4])<max_teta  ){
  			u[0]=st[0]/max_x;       // cart position: x                 
			u[1]=st[2]/max_teta;       // angle - pole 1
			u[2]=st[4]/max_teta;       // angle - pole 2
			output_nnet(u,z_old,z,y,sol);
			for (i=0;i<n_hid1;i++)
	            z_old[i]=z[i];	
	       
             if (i_out<0){
    			vote_sum=0.0;
    			for (i=0;i<n_out;i++){
    			    vote_sum=vote_sum+sign(y[i]);
    			    //vote_sum=vote_sum+w_com[i]*sign(y[i]);
                 }
                vote=sign_int(vote_sum);
        
                // comput output
                F=0.0;
                if (vote==0){
                    for (i=0;i<n_out;i++)
                        F=F+w_com[i]*y[i];
                }
                else {
                    w_com_sum=0.0;
                    for (i=0;i<n_out;i++){
                        if ( vote==sign_int(y[i]) ){
                           w_com_sum=w_com_sum+w_com[i];
                           F=F+y[i];
                        }
                    }                    
                    F=F/w_com_sum;
                }
            }
            else {
                F=y[i_out];  
            }
            F=F*10.0;	                 
			//cout<<" "<<F;
			Pole->step(F,st);				  // step of the inverted double pole dynamical system
      		// compute f_cheap
			fst[t]=fabs(st[0])+fabs(st[1])+fabs(st[2])+fabs(st[3]);
			if (t<=100){
				fit_stable=0.0;
				sum_fst=sum_fst+fst[t];
			}
			else{
				sum_fst=sum_fst+fst[t]-fst[t-101];
				fit_stable=0.75/sum_fst;
			}
			f_cheap=0.0001*t +0.9*fit_stable;
			//Pole->print();		
	    	t++;
	}
//	cout<<"t-> "<<t<<endl;
	fit=f_cheap;
  
	delete Pole;
	delete [] y;
	delete [] z;
	delete [] z_init;
	delete [] z_old;
	delete [] fst;
    	
	return (fit);   
}

/******************************************************************************\
*	Evaluation of esn with sol. defined by j - Generalization Test		        *
\******************************************************************************/
double nnetCom::evalGen(int *sol, int i_out, int com_esem){
  	int i, j=0, t, vote, i0, i1, i2, i3;
	double st[6], max_teta, max_x, max_teta_init, max_x_init, max_dteta_init, max_dx_init,  fit=0.0, *z, *z_init, *z_old, *y, u[3], F,   w_com_sum, vote_sum;
	double init_range[]={0.05, 0.25, 0.5, 0.75, 0.95}; // for the initial setting of the system
	doublePole *Pole;		// declaring Inverted Double Pole class variable
	
	
	// Memory allocation
	z=aloc_vectord (n_hid1);
	z_init=aloc_vectord (n_hid1);
	z_old=aloc_vectord (n_hid1);
	y=aloc_vectord (n_out);


	
	// Maximum values
	max_teta=36*(2*pi)/360;		    // (rad) maximum value for the pole angles
	max_x=2.4;					// (m) maximum value for the cart position (track size)
	max_teta_init=3.6*(2*pi)/360;		    // (rad) maximum value for the pole angles
	max_x_init=2.16;					// (m) maximum value for the cart position (track size)  
    max_dteta_init=8.6*(2*pi)/360;              // (rad/s) maximum value for the angles -> used for the initial settings
    max_dx_init=1.35;              // (m/s) maximum value for the cart velocity -> used for the initial settings

	// Initial inputs for the NNet
	for (i=0;i<n_inp;i++)
		u[i]=0.0;
	// Initial hidden neuron activations
	for (i=0;i<n_hid1;i++)
		z_init[i]=0.0;
	
	// Running the NNet for 20 iterations with zero input
	for (t=0;t<20;t++){
		output_nnet(u,z_init,z,y,sol);
		for (i=0;i<n_hid1;i++)
			z_init[i]=z[i];		
	}
	
	for (i0=0;i0<5;i0++){
        for (i1=0;i1<5;i1++){
          for (i2=0;i2<5;i2++){
              for (i3=0;i3<5;i3++){      
                // Initialization of the evaluation
                // States (st)
                st[0]=-max_x_init+init_range[i0]*2.0*max_x_init;   // (m) cart position: x
                st[1]=-max_dx_init+init_range[i1]*2.0*max_dx_init; // (m/s) cart velocity: dx
                st[2]=-max_teta_init+init_range[i2]*2.0*max_teta_init;      // (rad) angle - pole 1
                st[3]=-max_dteta_init+init_range[i3]*2.0*max_dteta_init;    // (rad/s) ang. velocity - pole 1
                st[4]=0.0;	                // (rad) angle - pole 2
            	st[5]=0.0;                  // (rad/s) ang. velocity - pole 2
            	t=0;						// iteration
            	for (i=0;i<n_hid1;i++)
            		z_old[i]=z_init[i];	
                   
            
            	Pole = new doublePole(st);	// initialization Inverted Double Pole
            		
            	// Simulation of the Double Pole
            	while(t<t_max_gen && fabs(st[0])<max_x && fabs(st[2])<max_teta && fabs(st[4])<max_teta  ){
              			u[0]=st[0]/max_x;       // cart position: x                 
            			u[1]=st[2]/max_teta;       // angle - pole 1
            			u[2]=st[4]/max_teta;       // angle - pole 2
            			output_nnet(u,z_old,z,y,sol);
            			for (i=0;i<n_hid1;i++)
            	            z_old[i]=z[i];	
            	        if (i_out<0){  
                            if (com_esem==0){
                                 vote=0;
                            }    
                            else {        
                    			vote_sum=0.0;
                    			for (i=0;i<n_out;i++){
                    			    vote_sum=vote_sum+sign(y[i]);
                    			    //vote_sum=vote_sum+w_com[i]*sign(y[i]);
                                 }
                                vote=sign_int(vote_sum);
                           }
                                
                            // comput output                            
                            F=0.0;
                            if (vote==0){
                                for (i=0;i<n_out;i++)
                                    F=F+w_com[i]*y[i];
                            }
                            else {
                                w_com_sum=0.0;
                                for (i=0;i<n_out;i++){
                                    if ( vote==sign_int(y[i]) ){
                                       w_com_sum=w_com_sum+1.0;
                                       F=F+y[i];
                                    }
                                }                    
                                F=F/w_com_sum;
                            }
                        }
                        else{
                            F=y[i_out]; 
                        }    
                        F=F*10.0;	                 
            			//cout<<" "<<F;
            			Pole->step(F,st);				  // step of the inverted double pole dynamical system
                  		//Pole->print();		
            	    	t++;
            	}
            //	cout<<"t-> "<<t<<endl;
            	if (t==t_max_gen){
                      fit=fit+1.0;
                      res_test[j]=1;
                 }
                 else 
                      res_test[j]=0;  
                 j++;    
                 
              
            	delete Pole;
            	
             }	
          }
        }
     }
             
    	
	delete [] y;
	delete [] z;
	delete [] z_init;
	delete [] z_old;
    	
	return (fit);   
}


/******************************************************************************\
*								Print NNet information														   *
\******************************************************************************/
void nnetCom::print(void){
	int i, j;
	
	cout<< "Echo State Network: "<<endl;
	cout<<" Number of Inputs: "<<n_inp<<endl;
	cout<<" Number of Neurons in Hidden Layer 1 (reservoir): "<<n_hid1<<endl;
	cout<<" Number of Neurons in Hidden Layer 2: "<<n_hid2<<endl;
	cout<<" Number of Outputs: "<<n_out<<endl;
	cout<<" Hidden Layer 1: "<<endl;
	for (i=0;i<n_hid1;i++){
			cout<<" Neuron: " <<i<<" , bias: "<< hid1_neurons[i].bias<<endl;
			cout<<"  W_in: ";	
			for (j=0;j<n_inp;j++)
					cout<< W_in[i][j] << ", ";
			cout<<endl;
			cout<<"  W: ";	
			for (j=0;j<n_hid1;j++)
					cout<< W[i][j] << ", ";
			cout<<endl;
	}
	cout<<" Hidden Layer 2: "<<endl;
	for (i=0;i<n_hid2;i++){
			cout<<" Neuron: " <<i<<" , bias: "<< hid2_neurons[i].bias<<endl;
			cout<<"  W_hid: ";	
			for (j=0;j<n_hid1;j++)
					cout<< W_hid[i][j] << ", ";
			cout<<endl;	
	}
	cout<<" Output Layer: "<<endl;
	for (i=0;i<n_out;i++){
			cout<<" Neuron: " <<i<<" , bias: "<< out_neurons[i].bias<<endl;
			cout<<"  W_out: ";	
			for (j=0;j<n_hid2;j++)
					cout<< W_out[i][j] << ", ";
			cout<<endl;
	}

	
}
