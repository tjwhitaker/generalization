/////////////////////////////////////////////////////////////////////////    
//  Cooperative Synapse NeuroEvolution (CoSyNE) 1.1 (beta)
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


#include <typeinfo>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::flush;
#include <fstream>
using std::ifstream;
using std::ofstream;
#include <unistd.h>
#include <time.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <numeric>

#include "TestNets.h"
#include "Environment.h"
#include "Neuron.h"
#include "Network.h"
#include "signal-handler.h"


#define SANE_MODE false
//#define _GUILE

#ifdef _XML
#include "CoSyNE_SAX_Handlers.h"
#endif

#include "CoSyNE.h"

extern void reseed(int val);

//////////////////////////////////////////////////////////////////////
//
// CoSyNE
//
//////////////////////////////////////////////////////////////////////

CoSyNE::CoSyNE(struct cosyneArgs *args, Environment &e, Network &n)
  : NeuroEvolution(e),
    verbose(args->verbose),  
    numUnits(args->numpops),         //!< set number of subpops.
    generation(0),                  //!< start at generation 0.
    prevBest(0.0),
    initWeightPopSize(args->init_popsize),
    perfQ(),                        //!< initial perfQ to size 0.
    averageFitness(0.0),            //!< set ave fitness to 0.
    stagnation(args->stagnation),   //!< set stagnation criteria.
    recombineP(true),               //!< whether to reombine = yes. 
    solved(false),
    seed(args->seed),
    sameSeed(args->same_seed)
  
{
  setParams(args);

}

//to do: check here that all type values are valid
void CoSyNE::setParams(struct cosyneArgs *args)
{
  mutationRate = args->mutation;  //!< set mutation rate.
  neutralMutationRate = args->neutral;   //!< set mutation rate.
  netType = intToNetEnum(args->nettype);
  subPopSize = args->popsize;      //!< set subpop size.
  numEvals = args->numevals;
  initWeightPopSize = args->init_popsize;
  stagnation = args->stagnation;   //!< set stagnation criteria.
  verbose = args->verbose;
  growNets = args->grownets;       //!< whether to grow the nets.
  saveNets = args->savenets;       //!< whether to save each new best net
  wtRange = args->wtrange;
  crossoverType = args->crossover_type;
  network = genNet(netType,0,0,0);
  
  //Weight prob params
  proportional  = args->proportional; 
  shift = args->shift; 
  creditAssignment = BEST_FITNESS;
}

//----------------------------------------------------------------------
//! initialize the neuron subpopulations 
void CoSyNE::init()
{
    //NOTE FIX THIS
    //if(args->seednet) numPops = network->getNumNeurons();
    setGeneSize();                  //!< calculate the input layer size and gene size

    //KEY CHANGE
    numPops = numUnits * geneSize;
    //!< allocate the subpopulations.


    create();
    
    phaseBest = genNetwork();   //!< generate net for best during burst mutation phase
    //    phaseBest->setEnvPtr(envt)
    bestNetwork = genNetwork(); //!< generate net for overall best
    solutionNet = genNetwork();    
    phaseBest->create();           
    bestNetwork->create();    
    solutionNet->create();
    

    //!< output a list of the parameter settings.
    echoParams();

    //Envt.init();
}

//----------------------------------------------------------------------
// destructor
CoSyNE::~CoSyNE()
{
  int i, j;
  for (i = 0; i < numPops; ++i) 
    for (j = 0; j < weightPops[i].size(); ++j) 
      delete weightPops[i][j];
  delete &Envt;
}
  

//! Print out the parameters
void CoSyNE::echoParams()
{   
  cout << "\n#CoSyNE settings:\n";
  cout << "#---------------\n";
  cout << "#Number of subpopulations : " << numPops           << endl;
  cout << "#Number of hidden units   : " << numUnits           << endl;
  cout << "#Size of subpopulations   : " << subPopSize        << endl;
  cout << "#Type of neural network   : " << network->getName() << endl;
  cout << "#Number of evals/neuron   : " << numEvals         << endl;
  cout << "#Crossover type           : ";
  switch(crossoverType){
  case 1:
    cout << "1-point" << endl;
    break;
  case 2:
    cout << "SBX" << endl;
    break;
  case 3:
    cout << "NO CROSSOVER" << endl;
    break;
  default:
    cerr << "unknown crossover type" << endl;
  }
  cout << "#Mutation rate            : " << mutationRate      << endl;
  if(neutralMutationRate > 0)
    cout << "#Neutral mutation rate            : " << neutralMutationRate      << endl;
  cout << "#Stagnation               : " << stagnation        << endl;  
  cout << "#Weight range             : [" << -wtRange << "," << wtRange << "]"    << endl;
  cout << "#Loci per weight          : " << NUM_LOCI        << endl;
  if(!growNets)
    cout << "#Adding/removing subpops  : DISABLED\n";
  if(minimize)
    cout << "#Fitness is being MINIMIZED" << endl;
  if(proportional){
    cout << "#Fitness proportional     : " << proportional << endl;
    cout << "#Shift                    : " << shift << endl;
  }
  if(sameSeed)
    cout << "#All evaluations use the SAME seed" << endl;
    
}


void CoSyNE::setNumPops(int n)
{ 
  if(n > 0 && n != numPops){
    numPops = n;
    setGeneSize();
    
  }
} 

void CoSyNE::setWeightPopSize(int n) { 
    if(n > 2 && n != subPopSize){ 
      subPopSize = n; 
      cout << "WeightPop size: " << subPopSize << endl; 
    }
}

//----------------------------------------------------------------------
//! Set the length of neuron chromosomes.
/*! Get the size of the Neuron chromosomes from the network
   and assign it to \c geneSize.  Need to get this from 
   a network because it will vary depending of the size and 
   type of networks being used.
*/
void CoSyNE::setGeneSize()
{
  Network *tmp = genNetwork();
  geneSize = tmp->getGeneSize();
  delete tmp;
}

//----------------------------------------------------------------------
/*! Create the subpopulations of neurons, initializing them to 
  random weights.
*/
void CoSyNE::create() 
{
  int i,j;
  
  for (i=0;i<numPops;++i){
    weightPops.push_back( vector<Weight*>(subPopSize) );// new std::vector<Weight*>(subPopSize);
    for(j=0; j < subPopSize; ++j) {
      weightPops[i][j] = new Weight();
      weightPops[i][j]->w = (drand48() - 0.5) * wtRange * 2.0;
    }
  }

}

//! Evolve the weight subpops.
/*!

  Evolve is the main genetic function.  The subpopulations are first
  evaluated in the given task.  Then for each subpop the neurons are
  ranked (sorted by fitness) and recombined using crossover.  
  Mating is allowed only between the top 'numBreed' members of 
  the same subpop. The neurons are then mutated.
*/

void CoSyNE::evolve(int gens, int evals)
{
  maxEvaluations = evals;
  stopEvolution = false;

  if(generation == 0) cout << "\nEvolving for " << gens << " generations.\n\n";
  gens += generation; //in case we are restarting
  while(!gInterrupt && generation++ < gens && !stopEvolution){
    //sprintf(outfile, "/tmp/%u_%d.esppop", getpid(), generation);
    //savePopulation(outfile);
    if (generation > 1) setWeightPopSize(subPopSize);
    
    evalPop();       //!< build and evaluate networks    

    //skip recombination if we have just burst mutated
    if(!recombineP) recombineP = true; 
    else {     
      recombine();     
    }
  }
#ifdef _GUILE
  --generation;
#else
  endEvolution();
#endif
}

//! Recombine the subpopulations.
/*! First sort the each of the of the subpopulations by fitness.
    Then take the top \c numBreed neurons in each subpop and 
    mate then with each other to produce enough neurons to 
    replace the bottom half of each subpop.  There is no
    mating between subpops.
*/

/*
 struct max_wt_fit : public std::binary_function<Weight*, Weight*, bool> {
    bool operator()(Weight *x, Weight *y) { return x->fitness->fitnesses[0][0] > y->fitness->fitnesses[0][0]; }
  }; 
*/

void CoSyNE::recombine()
{
  int i,j,k,mate,excess,numBreed;

 
  for (i = 0; i < numPops; ++i){    
    switch(creditAssignment){
    case BEST_FITNESS:
      sort(weightPops[i].begin(), weightPops[i].end(), max_best_fit() );
      break;  
    case AVERAGE_FITNESS:
      sort(weightPops[i].begin(), weightPops[i].end(), max_average_fit() );
      break;
    }
   
    if(minimize)
      reverse(weightPops[i].begin(), weightPops[i].end() );
  }
  // remove excess
  for (i = 0; i < numPops; ++i) {
    //get rid of excess neurons that have been added by net-level
    excess = weightPops[i].size() - subPopSize;
    for (j = 0; j < excess; ++j){
      delete weightPops[i][weightPops[i].size()-1];
      weightPops[i].pop_back();
    }
  }
  
  //parents
  Neuron *p1 = new Neuron(geneSize);
  Neuron *p2 = new Neuron(geneSize);
  //children
  Neuron *c1 = new Neuron(geneSize);
  Neuron *c2 = new Neuron(geneSize);
  p1->create();
  p2->create();
  c1->create();
  c2->create();
  

  Weight *weight;

  //crossover the best networks to get new weights
  for (i = 0; i < numUnits; ++i) {
    //    assert(weightPops[i]->size() == subPopSize);
    //!< \todo fix this to use Population::selectRndIndividual.
    numBreed = (int) subPopSize/4;
    
    if (crossoverType == 3) reseed(seed+generation);
    for (j = 0; j < numBreed; ++j){
      if(!j)  mate = lrand48() % numBreed;
      else mate = lrand48() % j;   
      //      cout << mate << endl;
      //load the weights into neurons
      for (k = 0; k < geneSize; ++k){	
	p1->setWeight(k, weightPops[i*geneSize+k][j]->w );
	p2->setWeight(k, weightPops[i*geneSize+k][mate]->w );
      }	
      //cross them
      switch(crossoverType)
	{
	case 1:
	  crossoverOnePoint(p1, p2, c1, c2);
	  break;
	case 2:
	  crossoverSBX(p1, p2, c1, c2);
	  break;
	case 3: //no crossover
	  *c1 = *p1;
	  *c2 = *p2;
	  break;
	default:
	  cerr << "unknown crossover type" << crossoverType << endl;
	}

              
 
      //put children's weights back in weight subpops
      // int tmp1 = lrand48() % subPopSize;
      //int tmp2 = lrand48() % subPopSize;
      for (k = 0; k < geneSize; ++k){
	weight = new Weight();
	weight->w =  c1->getWeight(k);
	weightPops[i*geneSize+k].push_back( weight );
	////weightPops[i*geneSize+k].insert(weightPops[i*geneSize+k].begin()+tmp1, weight );
	//weightPops[i*geneSize+k][tmp1]->w = c1->getWeight(k);
	weight = new Weight();
	weight->w =  c2->getWeight(k);
	weightPops[i*geneSize+k].push_back( weight );
	////weightPops[i*geneSize+k].insert(weightPops[i*geneSize+k].begin()+tmp2, weight );
	//weightPops[i*geneSize+k][tmp2]->w = c2->getWeight(k);
      }
      
      /*      
	weightPops[i][subPopSize-(1+j*2)]->w = weightPops[i][j]->w + rndCauchy(mutationRate); 
	weightPops[i][subPopSize-(2+j*2)]->w = weightPops[i][j]->w + rndCauchy(mutationRate); 
      */
      //
      //}
    }
  }
  delete p1;
  delete p2;
  delete c1;
  delete c2;
  
     
    
  //!< mutate the neurons in each subpop.
  for (i = 0; i < numPops; ++i)
    for (j = numBreed; j < weightPops[i].size(); ++j)
      if(drand48() < mutationRate)
	weightPops[i][j]->w += rndCauchy(0.3);
  
}






/////////////////////////////////////////////////////////
double CoSyNE::assignProb(WeightPop &sp, int n){

  double prob;

  if(getFitness(sp[n]) == 0) //>= subPopSize-1) individual is new 
    prob = 0.0;    //nc =1.0
  else if(!proportional)
    prob = 1.0;
  else{ 
    prob = ((getFitness(sp[n]) - getFitness(sp[subPopSize-1]) ) /
    	    (getFitness(sp[0]) - getFitness(sp[subPopSize-1]) ) ); //fitness proportional
    prob = (1-prob);// /numPops;
    prob += shift;
    prob *= 1/log(numPops);
    if(prob < 0) prob = 0;
    if(prob > 1.0) prob = 1.0;
    
    //cout << n << " " << prob << endl;
  }
   
  sp[n]->p = prob;
  //cout << n << " " << prob << endl;
  return prob;
  
}


//////////////////////////////////////////////////////////////////
void CoSyNE::getNetworks(vector<Network*> &nets)
{
  int i,j,k,w;
  vector< WeightPop > tmpPop(numPops);
  vector<int> log;
  vector< vector<int> > origLog(numPops);
  unsigned int num = weightPops[0].size();
  vector<int> log2(num,0);

  for(i = 0; i < numEvals; ++i){
    for(j = 0; j < numPops; ++j){
      log.clear();
      origLog[j].clear();
      tmpPop[j].clear();
      
      for(k = 0 ; k < num; ++k) {
	if( drand48() < assignProb(weightPops[j],k) ){
	  //cout << weightPops[j][k]->fitness << endl;
	  log2[k]++;
	  log.push_back(k);
	  tmpPop[j].push_back(weightPops[j][k]);
	}
	//if(j == 0) 	cout << weightPops[j][k]->p << " " << endl;
      }
      
      // cout << (double) log.size()/num << " ";
      //      permute(tmp);
      origLog[j] = log;
      
      permute(log);
      
      
      
      //permute only those selected
      for(k = 0 ; k < log.size(); ++k) {
	weightPops[j][log[k]] = tmpPop[j][k];
      }
    }
    /*    for(k = 0 ; k < num; ++k) 
      cout << log2[k] << " ";
    cout << endl;
    */
    ///////////////////////////////////////////  
    for(k = 0 ; k < num; ++k) {
      for(j = 0; j < numUnits; ++j){
	nets[i*num+k]->setNeuron( new Neuron(geneSize), j);
	//nets[i*num+k]->getNeuron(j)->create();
	//NOTE
  	for(w = 0; w < geneSize; ++w)
	  nets[i*num+k]->getNeuron(j)->setWeight(w, weightPops[j*geneSize+w][k]);
	nets[i*num+k]->getNeuron(j)->created = true; //!< \todo make automatic
      }
      nets[i*num+k]->created = true; //!< \todo make automatic

    }
  }
  
  //restore original order in weight pops. WHY? Maybe for later doing neuron level
  // or for fitness assignment
  /* for(j = 0; j < numPops; ++j)
    for(k = 0 ; k < origLog[j].size(); ++k) 
      weightPops[j][origLog[j][k]] = tmpPop[j][k];
  */
  // printFitnesses();
}


//--------------------------------------------------------------------
//! Evaluate the networks on the task.
/*!
  Evaluation stage.  Evaluate \c numTrials networks, each containing
  \c numPops neurons selected randomly from the subpopulations.  Each 
  participating neuron receives the fitness value of the 
  network it parcipates in.
  \todo rewrite this.
*/

void CoSyNE::evalPop()
{
  int i,j, numTrials;
  double newAveFit = 0;
  Network *bestNet = genNetwork(); 
  //  bestNet->created = true;
  bestNet->create();
  if(minimize)
    bestNet->setFitness(1000000);
 
  numTrials = weightPops[0].size() * numEvals;//NOTE

  vector<Network*> nets(numTrials);  


  // Build the networks
  for(i = 0; i < numTrials; ++i) {
    nets[i] = genNetwork();
    //    nets[i]->create();
  }
  getNetworks(nets); 

  //need to move this after getNetworks as the fitnesses are needed
  // to calculate probability of being permutated
  // reset the weight fitnesses
  for (i = 0; i < numPops; ++i) 
    for (j = 0; j < weightPops[i].size(); ++j) 
      weightPops[i][j]->resetFitness();
    


  //cout << weightPops[0]->getNumIndividuals() << endl;
  for(i = 0; i < numTrials; ++i) {
    if(gInterrupt) break;

    
    //evaluate the network 
    if(sameSeed) reseed(seed);
    Envt.evaluateNetwork(nets[i]);
    if(sameSeed) reseed( time(NULL) );

      //stop is the max allowed number of evaluations has been reached
    if(maxEvaluations && evaluations >= maxEvaluations){
      stopEvolution = true;
      break;
    }

    if(verbose)
      cout << "fitness " << i << ": " << getFitness(nets[i]) << endl;
	     


        
    if( compareFitness(nets[i], bestNet) ){ 
      *bestNet = *nets[i];    
      
      //      cout << *bestNet << endl;
      //cout << *nets[i] << endl;
      //#endif  
      //  NOTE NOTE NOTE    
      
      if( compareFitness( bestNet, Envt.goalFitness ) )
	break;
    }
    
  }

  
  if(compareFitness( bestNet, phaseBest ) )
    *phaseBest = *bestNet; 
	 
  if(compareFitness( bestNet, bestNetwork ) ) 
    setBestNetwork( phaseBest );
  
  averageFitness = newAveFit/numTrials;
  
  perfQ.push_front( getFitness(phaseBest) );

  //  cout << getFitness(phaseBest) << " " <<  perfQ[stagnation-1] << endl;  
  // if performance stagnates, do something
  if( perfQ.size() >= stagnation && 
      getFitness( phaseBest ) <= perfQ[stagnation-1] )
    //compareFitnessReverse( phaseBest, perfQ[stagnation-1] ) )
    handleStagnation();
 
  //printf("%f %f\n", perfQ.front(), perfQ[stagnation-1] );
  cout << "gen " << generation 
       << ": best " << getFitness(bestNet)
       << ", task best " << getFitness(phaseBest)
       << ", overall best " << getFitness(bestNetwork)
       << "\n";

  //testNetwork(&Envt, bestNet, true, true);
  // release neurons and delete the networks


  int k;
  bool tmp2 = false;
  bool inserted = true; //NOTE 
  for(i = 0; i < nets.size(); ++i) {
    if(tmp2) inserted = true;
    nets[i]->releaseWeights();
    //delete nets[i];
    
  }  
    
  delete bestNet;

}
  
  
//----------------------------------------------------------------------
//! Make a decision about what to do when performace stagnates.
void CoSyNE::handleStagnation(){
  perfQ.clear();   


  if(getFitness(phaseBest) <= prevBest ){
    //    if (growNets && !removeWeightPop(phaseBest))
    // addWeightPop();
    if( Envt.getIncremental() ){
      solutionNet->resetFitness();
      setBestNetwork(solutionNet);
      prevBest = 0;
      cout << "#Using last solution" << endl;
      if(!Envt.simplifyTask() ) // if the task hasn't changed terminate
	stopEvolution = true;
    }
  }
  else  prevBest = getFitness(phaseBest);
  
  
  burstMutate();  
  phaseBest->resetFitness();
}
 
 
//----------------------------------------------------------------------
//! Burst mutate the subpops.
void CoSyNE::burstMutate()
{
  burstMutate(bestNetwork);
}


void CoSyNE::burstMutate(Network *net)
{
  
  cout << "#BURST MUTATION STARTED\n";
  recombineP = false;
  int j,k;
  //NOTE
  //  testNetwork(&Envt, net, false, true);
  for(int i=0; i < net->getNumNeurons(); ++i){
    for(j=0; j < geneSize; ++j){
      for(k=0; k < weightPops[i*geneSize+j].size() ; ++k){
	//    if(i < net->getNumNeurons() )
	weightPops[i*geneSize+j][k]->w = 
	  net->getNeuron(i)->getWeight(j) + rndCauchy(0.05); 
	weightPops[i*geneSize+j][k]->resetFitness(); //NOTE  could use the Pop level call here
      }
    }
  }
}


/*! \todo Need to make sure that net a phase best are the same size.
      This could be handled in the assigment operator.
*/
void CoSyNE::setBestNetwork(Network *net)
{
  char fname[100];
  
  *bestNetwork = *net; 

  sprintf(fname , "net%u-%d-%f", getpid(), generation, getFitness(bestNetwork) );
  if(saveNets) bestNetwork->saveXML(fname);
    if(minimize){
      if(getFitness(bestNetwork) <= Envt.goalFitness)
	solved = true;
    }
    else
      if(getFitness(bestNetwork) >= Envt.goalFitness)
	solved = true;

  if(solved){
    phaseBest->resetFitness();
    bestNetwork->resetFitness();
    perfQ.clear();
    cout << "#Task Completed" << endl;
    if( Envt.getIncremental() ){
      *solutionNet = *bestNetwork; 
      cout << "#Evals " << evaluations << endl;
      bestNetwork->saveXML(fname);
      prevBest = 0;
      
      if( Envt.nextTask() ){
	stopEvolution = true;
      }
      else
	solved = false;

      setGeneSize(); //in case nextTask changes the I/O dimensionality
    }
    else{     
      stopEvolution = true;
    }
  }
}

/////////////////////////////////////////////////////////
// I/O fns
void CoSyNE::generatePopulationFromNetwork(Network *net)
{
  if(!net->created){
    cerr << "no network to seed!" << endl;
    exit(1);
  }
  this->network = net;
  
  numUnits = net->getNumNeurons();
  
  /* need an ::uncreate
  if(created)
  for (int i = 0; i < subPops.size(); ++i)  //!< construct numPops no. of NeuronPops
  delete subPops[i];
  */
  
  weightPops.clear();
  init();
  burstMutate(net);
}
 
#ifndef _XML
void CoSyNE::loadFile(const char *fname)
{
  int type;
  ifstream popIn;
  
  popIn.open(fname);
  
  if(!popIn){
    cerr << "\nError - cannot open " << fname << " (in CoSyNE::loadPopulation)" << endl;
    exit(1);
  }

  cout << "Loading subpopulations from file: " << fname << endl;
  popIn >> type;
  popIn >> generation;
  cout << generation << endl;
  popIn >> numPops;
  cout << numPops << endl;
  popIn >> geneSize;
  cout << geneSize << endl;
  popIn >> subPopSize;
  cout << subPopSize << endl;
  
  weightPops.clear();

  string id = "";

  phaseBest = genNetwork();       //!< generate new net 
  bestNetwork = genNetwork();     //!< generate new net 
  phaseBest->create();            // create net            
  bestNetwork->create();    

  cout << "DONE" << endl;
  popIn.close();
} 

#else

//using Xerces-C++ SAX parser
void CoSyNE::loadFile(const char *fname)
{
  ifstream fin(fname);

  if(fin.fail()){
    cerr << "ERROR: Could not open file (CoSyNE::loadFile): " << fname << endl;
    exit(1);
  }

  // Initialize the XML4C2 system
  try{
    XMLPlatformUtils::Initialize();
  }
  catch (const XMLException& toCatch){
    //        XERCES_STD_QUALIFIER cerr << "Error during initialization! Message:\n"
    //    << StrX(toCatch.getMessage()) << XERCES_STD_QUALIFIER endl;
    XERCES_STD_QUALIFIER cerr << "ERROR" << XERCES_STD_QUALIFIER endl;
    //        return 1;
  }
  SAXParser* parser = new SAXParser;
  CoSyNE_SAXHandlers* handler = new CoSyNE_SAXHandlers(this);
  
  parser->setDocumentHandler(handler);
  parser->setErrorHandler(handler);
  //  parser->setDoValidation(true);
  cout << "Loading subpopulations from file: " << fname << endl;
  parser->parse(fname);

  echoParams();

  savePopulation("tmp");

  /*
  phaseBest = genNetwork();       //!< generate new net 
  bestNetwork = genNetwork();     //!< generate new net 
  phaseBest->create();            // create net            
  bestNetwork->create();    
  */
}

#endif


/*! Save the neuron subpopulations to a file as text
  \todo add saving of netlevel
*/
void CoSyNE::savePopulation(char *fname)
{
  
  int i,excess;
    
  ofstream popOut;
  
  popOut.open(fname);
 
  
  if(!popOut){
    cerr << "\nError - cannot open " << fname << " (in CoSyNE::savePopulation)" << endl;
    exit(1);
  }

  cout << "\nSaving population to " << fname << endl;
  
  popOut << "<?xml version=\"1.0\"?>" << endl;
  popOut << "<!DOCTYPE COSYNE SYSTEM \"cosyne.dtd\">" << endl;
  popOut << "<COSYNE>" << endl;
  popOut << "   <PREAMBLE>" << endl;
  popOut << "       <ENVIRONMENT>" << Envt.getName() << "</ENVIRONMENT>" << endl;
  popOut << "       <NET_TYPE>" << network->getName() << "</NET_TYPE>" << endl;
  popOut << "       <GENERATION>" << generation << "</GENERATION>" << endl;
  popOut << "       <NUM_POPS>" << numPops << "</NUM_POPS>" << endl;
  // popOut << "       <GENE_SIZE>" << geneSize << "</GENE_SIZE>" << endl;
  //popOut << "       <SUB_POP_SIZE>" << subPopSize << "</SUB_POP_SIZE>" << endl; 
  popOut << "   </PREAMBLE>" << endl;
  

  int j;  
  for (i = 0; i < numPops; ++i){
    sort(weightPops[i].begin(), weightPops[i].end(), max_best_fit() );
    excess = weightPops[i].size() - subPopSize;
    for (j = 0; j < excess; ++j)
      weightPops[i].pop_back();
    popOut << "<POPULATION size=\"" << subPopSize << "\">" << endl;
    for (j = 0; j < subPopSize; ++j)
      popOut << weightPops[i][j]->w << " ";
    popOut << "</POPULATION>" << endl;
  }
  
  popOut << "</COSYNE>" << endl;
  popOut.close();
} 

//-------------------------------------------------------------

void CoSyNE::endEvolution()
{
  char fname[100];
  char popfile[100];

  printStats();
  sprintf(fname , "net%u", getpid() );
  sprintf(popfile , "pop%u", getpid() );
  cout << "Best Network " << Envt.evaluateNetwork(bestNetwork)  << endl;

  bestNetwork->saveXML("the_net");
  savePopulation(popfile);
  
  cout << "BYE!\n";
  exit(0);
}


void CoSyNE::printStats()
{
  cout << "\n#Total number of network evaluations : " << evaluations << endl;
}


#define LESION_THRESHOLD 0.8
/*! \todo 
    May want to reset fitness here or automatically reset it
    when lesioned is set by lesion(int).  Make lesioned private to
    \a Network so that even subclasses cannot access is directly.
    Think about this because there are cases were you might want
    to repeatedly eval with a particular lesion.  Have lesion(int)
    check if int is different since last eval.
*/
int CoSyNE::lesionTest( Network *net )  
{
  int i,j,sp;
  double lesionFitness, max = 0.0, min = 1000000000;

  for(i=0; i < 1; ++i)
    Envt.evaluateNetwork(net);
  double ulfit = getFitness(net);  
  if(minimize) cout << "UNlesioned :" << 1.0/ulfit + 1.0 << endl;
  else cout << "UNlesioned :" << ulfit << endl;
  for(i=0; i < net->getNumNeurons(); ++i){
    net->resetFitness();
    net->getNeuron(i)->lesioned = true;
    for(j=0; j < 1; ++j)
      Envt.evaluateNetwork(net);
    lesionFitness = getFitness(net);
    if(minimize) cout << "lesion " << i << ":" << 1.0/lesionFitness + 1.0 << endl;
    else cout << "lesion " << i+1 << ":" << lesionFitness << endl;
    net->getNeuron(i)->lesioned = false;
    if(minimize){
      if(lesionFitness < min){
	min = lesionFitness;
	sp = i;
      }
    }
    else
      if(lesionFitness > max){
	max = lesionFitness;
	sp = i;
      }
  }
  if(minimize){
    if(min <= (ulfit * LESION_THRESHOLD) )
      return sp;
  }
  else if(max >= (ulfit * LESION_THRESHOLD) )
    return sp;
  else return -1;
}



void CoSyNE::printFitnesses()
{ 
  int i,j;
     
  for(j=0; j < weightPops[0].size(); ++j) {
    for (i=0;i<numPops;++i)
      cout << getFitness(weightPops[i][j]) << " " ;
    cout << endl;
  }
  cout << endl;
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
