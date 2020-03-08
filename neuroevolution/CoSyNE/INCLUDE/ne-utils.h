#ifndef _NE_UTILS_
#define _NE_UTILS_
 
#include <iostream>
using std::cerr;
#include "Neural.h"
#include <functional>
using std::binary_function;

//----------------------------------------------------------------------
// generate a random number form a cauchy distribution centered on zero.
#define PI 3.1415926535897931 
double rndCauchy(double coeff);

//permute a vector
template <class T>
void permute(vector<T> &v)
{
  unsigned int i, num = v.size();
  vector<T> tmp(0);
  int index;
  
  int preserve = 0;
  
  for(i = 0; i < num; ++i){
    if(i < preserve)
      index = i;
    else
      index = lrand48() % v.size();

    tmp.push_back( v[index] );
    v.erase(v.begin()+index);
  }
  v = tmp;
}

//binary_functions using by STL sort algorithm to sort neurons and networks
struct max_average_fit : public binary_function<Neural*, Neural*, bool> {
  bool operator()(Neural *x, Neural *y){
  int obj = 0;

  Fitness *xf = x->getFitness();
  Fitness *yf = y->getFitness();
  
  return xf->getAveFitness() > yf->getAveFitness();
  }
};  

struct max_best_fit : public binary_function<Neural*, Neural*, bool> {
  bool operator()(Neural *x, Neural *y)	{  
    int obj = 0;
    bool result;
    
    Fitness *xf = x->getFitness();
    Fitness *yf = y->getFitness();
    
    /*    if(xf->getMaxFitness == NULL ||
       yf->getMaxFitness == NULL)
      {
	cerr << "ERROR: Comparing unevaluated neural object" << endl;
	exit(1);
      }
    else

    */
      return  xf->getMaxFitness() > yf->getMaxFitness();
    
  }
};
#endif
