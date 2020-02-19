/*! \file NE.h
 */

#ifndef _NE_
#define _NE_

#include <string>
using std::string;
class Fitness;
class Neural;
class Neuron;
class Network;
class Environment;

enum FitnessMeasure { 
  AVERAGE_FITNESS,
  BEST_FITNESS
};

//! Base class for other neuroevolution algorithms
/*! Class NeuroEvolution is a virtual class that 
    describes the basic structure that derived neuroevolutionary
    algirthms must take, as well as implementing the genetic operators
    for recombining Networks and Neurons;
\todo make genopts methods of Networks and Neurons.
*/


#include <functional>
using std::binary_function;

class NeuroEvolution {

protected: 
  int inputDimension;  //!< The number of variables that the nets receive as input.
  int outputDimension; //!< The number of variables in the action space.
  int evaluations;     //!< The number of network evaluations.
  FitnessMeasure creditAssignment; //!< How multiple evals a combined to measure fitness.
 
public:

  bool minimize;       //!< Whether or not fitness is maximized or minimized.
  string name;         //!< Name of the actual NE alg e.g. ESP
  Environment &Envt;   //!< The task environment.
  NeuroEvolution(Environment &e);   
  virtual ~NeuroEvolution() {}
  double getFitness(Neural *neural, int obj=0);
  double getFitness(Fitness *fit, int obj=0);
  bool compareFitness(Neural *, double);
  bool compareFitness(Neural *, Neural *);
  bool compareFitnessReverse(Neural *, double);
  // Accessors
  //  int getInDim() { return inputDimension; }
  //int getOutDim() { return outputDimension; }
  int getEvaluations() { return evaluations; }
  // Genetic operators
  void crossoverOnePoint(Neuron *, Neuron *, Neuron *, Neuron *);
  //NOTE
  void crossoverNormAverage(Neuron *parent1, 
			    Neuron *parent2, 
			    Neuron *child1, 
			    Neuron *child2);
  void crossoverSBX(Neuron *, Neuron *, Neuron *, Neuron *);
  void crossoverArithmetic(Neuron *, Neuron *, Neuron *, Neuron *);
  void crossoverEir(Neuron *, Neuron *, Neuron *, Neuron *);
  void crossoverOnePoint(Network *, Network *, Network *, Network *);
  void crossoverOnePointPermute(Network *, Network *, Network *, Network *);
  void crossoverArithmetic(Network *, Network *, Network *, Network *);
  void crossoverNPoint(Network *, Network *, Network *, Network *);
  void crossoverSBX(Network *, Network *, Network *, Network *);
  void incEvals() { ++evaluations; }
  virtual Network* genNetwork() = 0;
  //!< \todo make genNetwork() an abstract virtual of NE
  //maybe just make this bestNetwork  

};


#endif
