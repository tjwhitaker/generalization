/////////////////////////////////////////////////////////////////////////    
//  Cooperative Synapse NeuroEvolution (CoSyNE) 1.0 (beta) definitions
//  Copyright (C) 2007  Faustino J. Gomez
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//////////////////////////////////////////////////////////////////////////



/*! \file CoSyNE.h
    \brief Header file for CoSyNE.C
*/

#ifndef _CoSyNE_H_
#define _CoSyNE_H_

#include <fstream>
#include <vector>
#include <deque>
//#include <string>
#include "NE.h"
#include "Weight.h"

class Neuron;
class NetPop;
class Environment;
class Network;


//! User defined parameters for CoSyNE
struct cosyneArgs 
{
  int numpops;
  int popsize;
  int init_popsize;
  int numevals;
  Network *net;
  int seed; 
  int stagnation;  
  int nettype;
  bool netlevel; 
  int netlevelsize;
  bool grownets;
  double mutation;
  bool verbose;
  bool seednet;
  char netfile[100]; 
  std::string paramfile;
  bool savenets;
  bool perc;
  int loci;
  double neutral;
  bool proportional;
  double shift;
  double wtrange;
  int crossover_type;
  bool same_seed;
};


//NOTE this is the main difference from ESP
typedef std::vector<Weight*> WeightPop;



//! CoSyNE class
class CoSyNE : public NeuroEvolution {
  friend class CoSyNE_SAXHandlers;
private:
  std::vector< WeightPop > weightPops;  //!< std::vector of points to the subpopulations.
  // struct max_wt_fit : public std::binary_function<Weight*, Weight*, bool> {
  // bool operator()(Weight *x, Weight *y) { return x->fitness > y->fitness; }
  //}; 

public:
  CoSyNE(struct cosyneArgs *, Environment &, Network &);
  ~CoSyNE();

  Network* genNetwork() { return network->newNetwork(inputDimension, 
						    numUnits, 
						    outputDimension); }
  int getMinUnits() { Network *n = genNetwork(); 
                     int tmp = n->getMinUnits(); delete n; return tmp;}
  void setParams(struct cosyneArgs *);
  void echoParams();
  void init();
  void create(); // creates a random population of Neurons
  void evolve(int, int);
  void recombine();
  void endEvolution();
  void setMutation(double m) { mutationRate = m; }
  void activateBurstMutation();
  double getAveFit() { return averageFitness; }

  // Accessors

  int getNumPops() { return numPops; }
  EspNetType getNetType(){ return netType; }
  void setNumPops(int);
  void setWeightPopSize(int);
  int getGeneSize() { return geneSize; }
  void loadFile(const char *);
  void savePopulation(char *);
  //  Neuron* getNeuron(int i, int j) { return subPops[i]->getIndividual(j); }
  int getWeightPopSize() { return subPopSize; }
  int getGeneration() { return generation; }
  void reset();
  void generatePopulationFromNetwork(Network *);
  double assignProb(WeightPop &, int);
  bool verbose;


private:
  Network *network;
  double mutationRate;  //!< Rate of mutation.
  double neutralMutationRate;  //!< Rate of neutral mutation.

  EspNetType netType;
  int numPops;          //!< Number of subpopulations
  int numUnits;         //!< The number of units in the network.
  int subPopSize;       //!< Number of neurons in each subpopulation.
  int numEvals;
  int generation;       //!< Counter for the number of generations.
  double prevBest;
  int initWeightPopSize;
  double wtRange;
  int crossoverType;
  bool withESP;

  ///////////////////////////
  // Weight Level prob params
  bool proportional;
  double shift;
  ///////////////////////////

  //  bool forceBurstMutation;
  std::deque<double> perfQ;   //!< A queue of the best fitness from each gen.
  double averageFitness; //!< The average fitness for a generation.
  int geneSize;          //!< The length of the neuron chromosomes. \todo move to NE?
  unsigned int stagnation;        //!< Then number of generations w/out improvement before action is taken.
  bool recombineP;       //!< Whether to recombine. \todo change name
  int maxEvaluations;

  bool growNets;
  bool saveNets;
  bool stopEvolution;
  bool solved;
  int seed;               //!< the seed for the random number generator
  bool sameSeed;

  Network *bestNetwork; //!< \todo make private after debug
  Network *phaseBest; 
  Network *solutionNet;
  NetPop *networkLevel;

  CoSyNE(const CoSyNE &); // just for safety
  CoSyNE &operator=(const CoSyNE &);

  void evalPop();
  void getNetworks(std::vector<Network*> &);
  void setGeneSize();
  void setBestNetwork(Network *);
  void handleStagnation();
  void burstMutate();  
  void burstMutate(Network *);
  void printStats();
  int lesionTest(Network *);
  void printFitnesses();
};



#endif

