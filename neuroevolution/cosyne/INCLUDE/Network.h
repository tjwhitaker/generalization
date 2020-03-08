#ifndef _NETWORK_H_
#define _NETWORK_H_
/*! \file Network.h
 */

#include <vector>
using std::vector;

#include <string>
#include <math.h>

#include "genNet.h"
#include "Environment.h"
#include "Fitness.h"
#include "Neural.h"

class Behavior {
 public:
  Behavior () : traj(), compressedLength(0) {}
  vector<double> traj;
  int compressedLength;
};

class Neuron;


//! Neural network base class
/*! Virtual class for neural networks consisting
    of a \c std::vector of \a Neurons that are connected
    through the implementation is an \a activation
    function in the derived classes.
*/
class Network : public Neural {
  //  friend class ESP_SAXHandlers;

protected:
  vector<double> activation;
  vector<Neuron*> hiddenUnits;
  

public:
  Behavior behavior; 
  bool created;
  double bias;

  //  static int numObjectives;

  Network(int, int, int);
  Network(const Network &n){}
  
  virtual ~Network();
  virtual Network* newNetwork(int, int, int) = 0;
  //  virtual Network* newNetwork(int, Neuron *n) = 0;
  virtual Network* clone() = 0;
  virtual void growNeuron( Neuron * ) = 0;
  virtual void shrinkNeuron( Neuron *, int ) = 0;
  virtual void addNeuron() = 0;
  virtual void removeNeuron(int) = 0;
  virtual void activate(vector<double> &, vector<double> &) = 0; 
  inline virtual int getMinUnits() { return 1; }


  void releaseNeurons();
  void releaseWeights();
  void deleteNeurons();
  void operator=(Network &n);
  bool operator==(Network &n);
  bool operator!=(Network &n);
  void create();
  //  Network(const Network &n) : subPop(n){;}
  virtual void resetActivation();
  void setNeuron(Neuron *, int);
  void swapNeurons(int, int);
  void setNetwork(Network *);
  void assignFitness(double fit, int obj = 0);
  void perturb(Network *);
  Network* perturb(double coeff = 0.3);
  void mutate(double);
  void printActivation(FILE *){}//GET RID OF THIS
  inline double getActivation(int i){ return activation[i];}
  inline void setActivation(int i, double d){activation[i] = d;}
  void saveXML(char *);
  void resetFitness() { fitness->reset(); }

  friend double Environment::evaluateNetwork(Network *);
  inline int getNumInputs() { return numInputs; }
  inline int getNumNeurons() { return (int) hiddenUnits.size(); }
  inline int getNumOutputs() { return numOutputs; }
  //Fitness* getFitness(){ return fitness;  }
  void setFitness (double fit, int obj = 0); //NOTE//<! make private again
  Neuron* getNeuron(int);
  inline int getGeneSize() { return geneSize; }
  int getParent (int);
  inline int getID() { return id; }
  inline void setParent(int p, int id)
  {
    if(p == 1)
      parent1 = id;
    else if(p == 2)
      parent2 = id;
    //<! \todo    else
    // cerr <<  "Parent must be 1 or 2; Network::getParent" << endl;
  }
  const char* getName() { return getNetTypeName(this); }
  enum EspNetType getType () { return netStrNameToEnum(getNetTypeName(this));}
  void lesion(int); 
  void unlesion(); 
  //void setEnvPtr(Environment *e) { envPtr = e; } 
  void setEnvPtr(Environment *e) { 
    envPtr = e; 
    if(envPtr->getNumObjectives() != fitness->getNumObjectives()){
      delete fitness;
      fitness = new Fitness(envPtr->getNumObjectives());
      //Fitness::numObjectives = envPtr->getNumObjectives();
    }
  } 
  Environment* getEnvPtr() { return envPtr; }
  //NOTE
  void makeWtVector();
  void mapBackToNeurons();
  /*
  friend void NeuroEvolution::crossoverOnePoint(Network *parent1, 
						Network *parent2, 
						Network *child1, 
						Network *child2);
  friend void NeuroEvolution::crossoverNPoint(Network *parent1, 
					      Network *parent2, 
					      Network *child1, 
					      Network *child2);
  */

  //  double fitness; //!< \todo protect this
  // Fitness *fitness;

  //!< NOTE
  vector<double> weights;

protected:
  int trials; 

  int id;       //!< \todo make id a struct that has p1,p2,and name;
  int parent1;
  int parent2;
  int geneSize;
  //int type;
  bool frozen;
  int numInputs; 
  int numOutputs;

  Environment *envPtr; //pointer to the environment in which it is being evaluated
  void addConnection(int);
  void removeConnection(int);
  inline double sigmoid(double x, double slope = 1.0)
  {
    return (1/(1+exp(-(slope * x)) ) );
  }
  inline double tanh(double x)
  {
    return (exp(2 * x)-1)/(exp(2 * x)+1);
  }
private:
 
  bool sizeEqual(Network &n);
};


std::ostream& operator<<(std::ostream &, Network &);
std::ostream& operator<<(std::ostream& os, std::vector<double> vec);


#endif
