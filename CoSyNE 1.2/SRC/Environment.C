/////////////////////////////////////////////////////////////////////
//  Environment class for CoSyNE 1.0
//  Copyright (C) 2007  Faustino J. Gomez
//
/////////////////////////////////////////////////////////////////////

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include "NE.h"
#include "Fitness.h"
#include "Network.h"
#include "Environment.h"

extern int Fitness::numObjectives;

void Environment::setNumObjectives(int i) { 
  Fitness::numObjectives = numObjectives = i; }

//! Evaluate a \a Network in the Environment.
/*! Takes a \a Network and evaluates it on a task.  First
    checks to see if it is connected to a neuroevolution 
    algorithm (\a NE), then increments the algorithm's 
    evaluate the network, assigns it a fitness and return the
    fitness of the network.  This function is a friend of \a 
    Network and is the only function outside of the \a Network
    class that can set the value of a Network.  This ensures
    that networks are only assigned fitness when they are 
    evaluated.
*/
   
double Environment::evaluateNetwork(Network *net) //bool assignFitness)
{
  double fit;

  if(nePtr) nePtr->incEvals();  //<! increment the algorithms \c evaluations member.
  net->setEnvPtr(this);
  net->resetActivation();
  //<! make sure the network has the right size I/O layers for the environment
  if( net->getNumInputs() == inputDimension &&
      net->getNumOutputs() == outputDimension )
    fit = evalNet(net);
  else{
    cerr << "Network dimensionality does not match environment!" << endl;
    exit(2);
  }
  
  for(unsigned int i = 0; i < numObjectives; ++i)
    net->assignFitness( fit, i );

  
  return fit;
}

