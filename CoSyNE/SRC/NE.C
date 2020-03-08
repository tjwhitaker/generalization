#include <algorithm>
#include <limits>
#include "Environment.h"
#include "Neuron.h"
#include "Network.h"
#include "NE.h" 



NeuroEvolution::NeuroEvolution(Environment &e) 
  : inputDimension(e.getInputDimension() ),
    outputDimension(e.getOutputDimension() ),
    evaluations(0),  
    minimize(e.minimize),
    Envt(e)
{
  Envt.setNePtr(this);
}


double NeuroEvolution::getFitness(Neural *neural, int obj)
{
  return getFitness( neural->getFitness(), obj );
}

double NeuroEvolution::getFitness(Fitness *fit, int obj)
{
  double result;

  //if neural has been assigned a fitness
  if(fit->getNumTrials()){
    switch (creditAssignment)
      {
      case AVERAGE_FITNESS:
	result = fit->getAveFitness();
	break;
      case BEST_FITNESS:
	if(minimize)
	  result = fit->getMinFitness();
	else 
	  result = fit->getMaxFitness();
	break;
      }
  }
  else//if fitness not defined   
    if(minimize)
      result = std::numeric_limits<double>::max(); 
    else
      result = 0;
  
  return result;
}



/*! 
*/
//compare the fitness of a Neural:: object
//with some scalar: fit
//according to the creditAssignment mode and whether
//fitness is beign minimize/maximized
bool NeuroEvolution::compareFitness(Neural *n1, double fit)
{  
  bool result;
   
  result = getFitness(n1) > fit;
  
  if(minimize)
    result = !result;
 
  return result;
}

//compare the fitness of two Neural:: objects
//according to the creditAssignment mode and whether
//fitness is beign minimize/maximized
bool NeuroEvolution::compareFitness(Neural *n1, Neural *n2)
{
  return  compareFitness(n1, getFitness(n2));
}

bool NeuroEvolution::compareFitnessReverse(Neural *n1, double fit)
{
  return  !compareFitness(n1, fit);
}

//----------------------------------------------------------------------
//! Simulated Binary Crossover
// Deb and Agrawal 1995
void NeuroEvolution::crossoverSBX(Neuron *parent1, 
				  Neuron *parent2, 
				  Neuron *child1, 
				  Neuron *child2)
{
  double u = drand48();
  double beta, n;
  register unsigned int i;
  
  n = 20.0;
  if(u <= 0.5)
    beta = pow((2.0 * u), (1.0 / n + 1));
  else
    beta = pow((1.0/(2.0 * (1.0 - u))), (1.0 / n + 1));
  *child1 = *parent2; //just to make sure that children
  *child2 = *parent1; // have been created

  child1->parent1 = parent1->getID();
  child1->parent2 = parent2->getID();
  child2->parent1 = parent1->getID();
  child2->parent2 = parent2->getID();
  child1->resetFitness();
  child2->resetFitness();  
  for (i=0 ; i < parent1->getSize(); ++i) {
    child1->setWeight(i, 0.5 * ((1.0 + beta) * parent1->getWeight(i) + 
                                (1.0 - beta) * parent2->getWeight(i)) );
    child2->setWeight(i, 0.5 * ((1.0 - beta) * parent1->getWeight(i) + 
                                (1.0 + beta) * parent2->getWeight(i)) );
  }
}

//----------------------------------------------------------------------
//! Arithmetic crossover

void NeuroEvolution::crossoverArithmetic(Neuron *parent1, 
					 Neuron *parent2, 
					 Neuron *child1, 
					 Neuron *child2)
{
  double a,b;
  register unsigned int i;
  
  a=0.25;
  b=0.75;
  
  child1->parent1 = parent1->getID();
  child1->parent2 = parent2->getID();
  child2->parent1 = parent1->getID();
  child2->parent2 = parent2->getID();
  child1->resetFitness();
  child2->resetFitness();  
  for (i=0 ;i < parent1->getSize(); ++i) {
    child1->setWeight(i, a * parent1->getWeight(i) + (b * parent2->getWeight(i)) );
    child2->setWeight(i, a * parent2->getWeight(i) + (b * parent1->getWeight(i)) );
   }
}

//! Another linear combination crossover.
void NeuroEvolution::crossoverEir(Neuron *parent1, 
				  Neuron *parent2, 
				  Neuron *child1, 
				  Neuron *child2)
{
  int i;
  double d2,d = 0.4; //was int for some time (?)
  d2 = 2*d + 1;
  child1->parent1 = parent1->getID();
  child1->parent2 = parent2->getID();
  child2->parent1 = parent1->getID();
  child2->parent2 = parent2->getID();
  
  for ( i=0 ; i < (int) parent1->getSize() ; ++i ) {
    child1->setWeight(i, parent1->getWeight(i) + (((drand48() * d2) - d) *
					      (parent2->getWeight(i) - parent1->getWeight(i))) );
    child2->setWeight(i, parent2->getWeight(i) + (((drand48() * d2) - d) *
					      (parent1->getWeight(i) - parent2->getWeight(i))) );
  }

/*
  child1->p1 = parent1->sp;
  child1->p2 = parent2->sp;
  child2->p1 = parent1->sp;
  child2->p2 = parent2->sp;
  child1->sp = parent1->sp;
  child2->sp = parent1->sp;
*/
}


//! One-point crossover for neurons
/*!
    Two parent neurons are mated to produce two offspring 
    by exchanging chromosal substrings at a random crossover point.
*/

void NeuroEvolution::crossoverOnePoint(Neuron *parent1, 
				       Neuron *parent2, 
				       Neuron *child1, 
				       Neuron *child2)
{
  //find crossover point
  
  double tmp;
  //  int cross1 = lrand48() % (parent1->getSize()-1) + 1;
  int cross1 = lrand48() % (parent1->weight.size()-1) + 1;
  
  //  if(parent1->weight.size() > parent2->getSize())
  // cross1 = lrand48() % parent2->getSize();
  if(parent1->weight.size() > parent2->weight.size())
    cross1 = lrand48() % parent2->weight.size();
  *child1 = *parent2;
  *child2 = *parent1; 
     
  if(parent1->tag || parent2->tag){
    child1->tag = true;
    child2->tag = true;
  }
  
  child1->parent1 = parent1->getID();
  child1->parent2 = parent2->getID();
  child2->parent1 = parent1->getID();
  child2->parent2 = parent2->getID();
  
  child1->resetFitness();
  child2->resetFitness();
 
  for(int i = 0; i < cross1; ++i){
    //for(j = 0; j < NUM_LOCI; ++j){
    //tmp = child2->weight[i * NUM_LOCI+j];    
    //child2->weight[i * NUM_LOCI+j] =  child1->weight[i * NUM_LOCI+j];
    //child1->weight[i * NUM_LOCI+j] = tmp;
    tmp = child2->getAllele(i);
    child2->setAllele( i, child1->getAllele(i) );
    child1->setAllele( i, tmp );
    //   }
  }   
 

  //child1 = parent1->crossoverOnePoint(parent2);
  //child2 = parent2->crossoverOnePoint(parent1);
  //  swap_ranges(child1->weight.begin(), child1->weight.begin()+cross1, child2->weight.begin());
}


//NOTE: temporary
void NeuroEvolution::crossoverNormAverage(Neuron *parent1, 
					  Neuron *parent2, 
					  Neuron *child1, 
					  Neuron *child2)
{
  //find crossover point


  *child1 = *parent2;
  *child2 = *parent1;

  child1->parent1 = parent1->getID();
  child1->parent2 = parent2->getID();
  child2->parent1 = parent1->getID();
  child2->parent2 = parent2->getID();
  
  child1->resetFitness();
  child2->resetFitness();

  double length1 = 0, length2 = 0;

  for(unsigned int i = 0; i < parent1->getSize(); ++i){
    length1 += (parent1->getWeight(i) + parent2->getWeight(i)) *
              (parent1->getWeight(i) + parent2->getWeight(i));
    length2 += (parent1->getWeight(i) - parent2->getWeight(i)) *
                 (parent1->getWeight(i) - parent2->getWeight(i));
    child1->setWeight( i, parent1->getWeight(i) + parent2->getWeight(i));
    child2->setWeight( i, parent1->getWeight(i) - parent2->getWeight(i));
  }
  length1 = sqrt(length1);
  length2 = sqrt(length2);
  for(unsigned int i = 0; i < parent1->getSize(); ++i){
    child1->setWeight( i, child1->getWeight(i)/length1 );
    child2->setWeight( i, child2->getWeight(i)/length2 );
  }
}
/*

void NeuroEvolution::crossoverOnePoint( Neuron *parent1, 
				        Neuron *parent2, 
					Neuron *child1, 
					Neuron *child2)
{
  //find crossover point
  if(child1) delete child1;
  if(child2) delete child2;

  child1 = parent1->crossoverOnePoint(*parent2);
  child2 = parent2->crossoverOnePoint(*parent1);
}
*/

//! One-point crossover for Networks
/*!
    Two parent Networks are mated to produce two offspring 
    by exchanging chromosal substrings at a random crossover point.
*/
/*
void NeuroEvolution::crossoverOnePoint(Network *parent1, 
				       Network *parent2, 
				       Network *child1, 
				       Network *child2)
{
  int crossNeuron = lrand48()%parent1->getNumNeurons();
    
  *child1 = *parent1;
  *child2 = *parent2;
   if(parent1->getNumNeurons() > parent2->getNumNeurons())
    crossNeuron = lrand48()%parent2->getNumNeurons();
  child1->resetFitness();
  child2->resetFitness();
  crossoverOnePoint(parent1->getNeuron(crossNeuron),
		    parent2->getNeuron(crossNeuron),
		    child1->getNeuron(crossNeuron),
		    child2->getNeuron(crossNeuron));
  child1->setParent(1, parent1->getID() );
  child1->setParent(2, parent2->getID() );
  child2->setParent(1, parent1->getID() );
  child2->setParent(2, parent2->getID() );
}
*/

void NeuroEvolution::crossoverOnePoint(Network *parent1, 
				       Network *parent2, 
				       Network *child1, 
				       Network *child2)
{

  
  //int offset = parent1->getNumNeurons() * parent1->getNumInputs();
  //int crossPoint = offset + lrand48() % (parent1->weights.size() - offset);
  int crossPoint = lrand48() % (parent1->getNumNeurons() * parent1->getGeneSize());
    
  *child1 = *parent1;
  *child2 = *parent2;
 
  //NOTE
  //child1->makeWtVector();
  //child2->makeWtVector();

  child1->resetFitness();
  child2->resetFitness();
  child1->setParent(1, parent1->getID() );
  child1->setParent(2, parent2->getID() );
  child2->setParent(1, parent1->getID() );
  child2->setParent(2, parent2->getID() );
  int n = parent1->getNumNeurons(), g = parent1->getGeneSize();
  int j;
  //  double tmp;
  for(int i = 0; i < n; ++i){
    for(j = 0; j < g; ++j){
      if(i*g+j < crossPoint){
	child1->getNeuron(i)->setWeight(j, parent2->getNeuron(i)->getWeight(j) );
	child2->getNeuron(i)->setWeight(j, parent1->getNeuron(i)->getWeight(j) );
      }
    }
    //for(j = 0; j < NUM_LOCI; ++j){
    //tmp = child2->weight[i * NUM_LOCI+j];    
    //child2->weight[i * NUM_LOCI+j] =  child1->weight[i * NUM_LOCI+j];
    //child1->weight[i * NUM_LOCI+j] = tmp;
    
    
  }
      //child1->mapBackToNeurons();
      //child2->mapBackToNeurons();
}

void NeuroEvolution::crossoverOnePointPermute(Network *parent1, 
					      Network *parent2, 
					      Network *child1, 
					      Network *child2)
{
  int crossNeuron = lrand48()%parent1->getNumNeurons();
    
  *child1 = *parent1;
  *child2 = *parent2;
  
  if(parent1->getNumNeurons() > parent2->getNumNeurons())
    crossNeuron = lrand48()%parent2->getNumNeurons();

  
  child1->resetFitness();
  child2->resetFitness();
  crossoverOnePoint(parent1->getNeuron(crossNeuron),
		    parent2->getNeuron(crossNeuron),
		    child1->getNeuron(crossNeuron),
		    child2->getNeuron(crossNeuron));
  child1->setParent(1, parent1->getID() );
  child1->setParent(2, parent2->getID() );
  child2->setParent(1, parent1->getID() );
  child2->setParent(2, parent2->getID() );

  for(int i = 0; i < parent1->getNumNeurons(); ++i)
    if(*child1->getNeuron(i) == *child2->getNeuron(i)){
      //      child1->swapNeurons(i, lrand48()%parent1->getNumNeurons() );
      // child2->swapNeurons(i, lrand48()%parent2->getNumNeurons() );
      child1->perturb();
     
      //  break;
    }
}





//! N-point crossover for Networks
/*!
    Two parent Networks are mated to produce two offspring 
    by exchanging chromosal substrings at N random crossover points.
*/
void NeuroEvolution::crossoverNPoint(Network *parent1, 
				       Network *parent2, 
				       Network *child1, 
				       Network *child2)
{
  *child1 = *parent1;
  *child2 = *parent2;
  child1->resetFitness();
  child2->resetFitness();
  for(int i = 0; i < parent1->getNumNeurons(); ++i){
    crossoverOnePoint(parent1->getNeuron(i),
		      parent2->getNeuron(i),
		      child1->getNeuron(i),
		      child2->getNeuron(i));
  }
  child1->setParent(1, parent1->getID() );
  child1->setParent(2, parent2->getID() );
  child2->setParent(1, parent1->getID() );
  child2->setParent(2, parent2->getID() );
}

//check for size
void NeuroEvolution::crossoverArithmetic(Network *parent1, 
					 Network *parent2, 
					 Network *child1, 
					 Network *child2)
{
  *child1 = *parent1;
  *child2 = *parent2;
  child1->resetFitness();
  child2->resetFitness();
  for(int i = 0; i < parent1->getNumNeurons(); ++i)
    crossoverArithmetic(parent1->getNeuron(i), parent2->getNeuron(i),
			child1->getNeuron(i), child2->getNeuron(i) );
  child1->setParent(1, parent1->getID() );
  child1->setParent(2, parent2->getID() );
  child2->setParent(1, parent1->getID() );
  child2->setParent(2, parent2->getID() );
}

//check for size
void NeuroEvolution::crossoverSBX(Network *parent1, 
					 Network *parent2, 
					 Network *child1, 
					 Network *child2)
{
  *child1 = *parent1;
  *child2 = *parent2;
  child1->resetFitness();
  child2->resetFitness();
  for(int i = 0; i < parent1->getNumNeurons(); ++i)
    crossoverSBX(parent1->getNeuron(i), parent2->getNeuron(i),
			child1->getNeuron(i), child2->getNeuron(i) );
  child1->setParent(1, parent1->getID() );
  child1->setParent(2, parent2->getID() );
  child2->setParent(1, parent1->getID() );
  child2->setParent(2, parent2->getID() );
}
