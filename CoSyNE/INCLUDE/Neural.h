#ifndef _NEURAL_
#define _NEURAL_

#include "Fitness.h"
#include <iostream>
using std::cout;

class Neural {
public:
 Neural()//NeuroEvolution *_ne)
    : //nePtr(_ne),
  fitness( new Fitness() )
    {}
  virtual ~Neural(){delete fitness;}
  
  //virtual ~Neural(){ delete fitness; }
    
  virtual void create() = 0;
  virtual void resetFitness() = 0;
  Fitness* getFitness() { return fitness; }

 
protected:
  Fitness *fitness;
  //NeuronEvolution *nePtr;
};

#endif
