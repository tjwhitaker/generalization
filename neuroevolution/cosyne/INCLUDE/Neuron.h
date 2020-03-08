#ifndef _NEURON_
#define _NEURON_

/*! \file Neuron.h

  Multiple loci per weight 
*/

//#include "Neural.h"
#include <string>
using std::string;
#include <stdio.h>
#include <vector>
#include <iostream>
using std::endl;
using std::cout;
using std::cerr;
#include "ne-utils.h"
#include "Weight.h"
#include "Fitness.h"

//! Neuron Class
/*! Neuron base class.  A neuron is the basic unit used
   to construct neural networks. 
*/

extern int NUM_LOCI;

#include "Neural.h"

class Neuron : public Neural{
  
public:
  //private:   
  Neuron(int size, double wtrange = 12.0);
  virtual ~Neuron();//{} // cout << "Destroying " << id << endl; }
  virtual Neuron* clone() { return new Neuron( weight.size()/NUM_LOCI ); }  
  virtual Neuron& operator=(const Neuron &);
  bool operator==(Neuron &);
  bool operator!=(Neuron &);
  virtual void create();
  void assignFitness(double fit, int obj = 0);
  virtual void resetFitness() { fitness->reset(); }
  virtual void addConnection(int);
  virtual void removeConnection(int);
  void perturb(Neuron *, double coeff = 0.3);
  void perturb(Neuron *, double (*randFn)(double), double);
  Neuron* perturb(double coeff = 0.3);
  virtual void mutate();
  void mutate(double);
  void mutateNeutral(double);
  void split();
  //  Fitness* getFitness(){ return fitness; }

  void releaseWeights();    
  void setWeight(int, Weight *);
  void setWeight(int, double);
  void setAllele(int, double);
  void copyWeight(int, Neuron *);  
  inline bool checkBounds(int i) 
  { 
    if(i >=0 && i < (int) getSize()) return true; 
    else { cerr << "Error: weight index out of bounds" << endl; abort();
    }
  }
  inline unsigned int getSize() { 
    return weight.size()/NUM_LOCI; }
  inline double getWeight(int i) { 
    double result = 0;
    if( checkBounds(i) ){
      for(int j = 0 ; j < NUM_LOCI ; ++j)
	result += weight[i*NUM_LOCI+j]->w;
    }  
  return result;
  }
  inline double getAllele(int i) { if( i < weight.size() )  return weight[i]->w;}
  inline int getID() { return id; }
  inline std::string getName() { return name; }
  //  inline int getTrials() { return trials; }
  inline void addRank(int i){ rank.push_back(i); }
  inline int getRank(){ if(rank.empty()) return -1; else return rank.back(); }
  inline int getAge() { return (int) rank.size(); }
  Neuron* crossoverOnePoint (Neuron *);
  friend std::ostream& operator<<(std::ostream& os, Neuron &n);
  double getHashKey();
  void setFitness (double fit, int obj) { fitness->setFitness(fit, obj); }

public:
  bool lesioned;
  int parent1;
  int parent2;
  bool tag;
  double p;
  //PROTECT  
  vector<Weight*> weight; 
  bool created; //NOTE PROTECT

protected:
  inline int newID() { Neuron n(0); id = n.getID(); rank.clear(); return id; }

  //NEURAL Fitness *fitness;

  int id;
  string name;
  bool frozen;
  bool elite;
  std::vector<int> rank; 
  double wtRange;
  //  int numLoci;
};


#endif






