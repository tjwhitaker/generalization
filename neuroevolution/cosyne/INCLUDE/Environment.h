/////////////////////////////////////////////////////////////////////
//  Environment class for CoSyNE 1.0 (beta)
//  Copyright (C) 2007  Faustino J. Gomez
//
//  Virtual Class for modelling a specific environment/task.
//  The only function that really needs to do anything when implemented
//  is evalNet. It must take a Network and return a double.  That is, 
//  evalNet is not restricted in any way in what it does with the 
//  Network (except alter its weights) as long as it returns 
//  a double (fitness).
//
////////////////////////////////////////////////////////////////////

#ifndef _ENVT_
#define _ENVT_

/*! \file Environment.h
 */


class Network;

#include <string>
using std::string;
#include <vector>
using std::vector;
#include <cstdio>
#include "Fitness.h"

class NeuroEvolution;

/*! Virtual class that describes the interface  
    for all task environments used in \c NeuroEvolution
    objects.
*/
class Environment {
public:
  double goalFitness;  
  
  Environment() :  minimize(false),
		   nePtr(NULL),
		   tolerance(0), 
		   incremental(false),
                   numObjectives(1) { Fitness::numObjectives = numObjectives; } 

  virtual ~Environment() {}
  double evaluateNetwork(Network *net);
  //  virtual void init() = 0;
  virtual bool nextTask() { return false; }   
  virtual bool simplifyTask() { return false; }
  virtual double evalNetDump(Network *net, FILE *) { return 0.0;} 
  virtual double generalizationTest(Network *) { return 0.0;}
  void setObjective(int o) { objective = o; }
  void setNePtr(NeuroEvolution *e) { nePtr = e; }
  inline int getInputDimension() { return inputDimension; }
  inline int getOutputDimension() { return outputDimension; }
  inline double getTolerance() { return tolerance; }
  inline bool getIncremental() { return incremental; }
  inline string getName() { return name; }
  inline int getNumObjectives() { return numObjectives; }
  void setNumObjectives(int i);
  bool minimize;

protected:
  NeuroEvolution *nePtr;  //!< Pointer to the Neuroevolution algorithm 
  string name;
  double tolerance;
  bool incremental;
  int inputDimension;     //!< dimension of input space
  int outputDimension;    //!< dimension of output space 
  int objective;

  virtual void setupInput(vector<double> &input){}
  virtual double evalNet(Network *net) = 0;   // evaluate a network

private:
  int numObjectives;
};

#endif


