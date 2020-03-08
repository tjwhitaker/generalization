//////////////////////////////////////////////////////////////////////
//
// Network
//
//////////////////////////////////////////////////////////////////////

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <fstream>
using std::ofstream;

#include "Network.h" 
#include "Neuron.h"
 

double rndCauchy(double wtrange);



std::ostream& operator<<(std::ostream& os, Network &net)
{
  os << "<NETWORK type=\"" << net.getName() << "\" "
     << "inputs=\"" << net.getNumInputs() << "\" "
     << "size=\"" << net.getNumNeurons() << "\" "
     << "outputs=\"" << net.getNumOutputs() << "\""
     << ">" << endl;
  if(net.getEnvPtr())
    os << "<ENVIRONMENT>" << net.getEnvPtr()->getName() << "</ENVIRONMENT>" << endl;
  for(int i=0; i <  net.getNumNeurons(); ++i)
    os << *net.getNeuron(i) << endl;
  os << "<FITNESS>" << net.getFitness() << "</FITNESS>" << endl;
  os << "</NETWORK>" << endl;
  return os;
}

std::ostream& operator<<(std::ostream& os, std::vector<double> vec)
{
  for(unsigned int i=0; i < vec.size(); ++i)
     os << vec[i] << " ";
  os << endl;
}


//////////////////////////////////////////////////////////////
// Network
//
Network::Network(int in, int hid, int out)
 : activation(hid), 
   hiddenUnits(hid),
   created (false),
   numInputs(in),
   numOutputs(out),
   bias(0.0), 
   trials(0),
   //   fitness(new Fitness()), //param not necessary?
   parent1(-1),
   parent2(-1),
   frozen(false),
   envPtr(NULL),
   weights(0)
   
{
  behavior.compressedLength = 0;
  static int counter = 0;
  id = ++counter;  
}


Network::~Network()
{
  
  //  behavior.traj.clear();
  deleteNeurons();
}


//! delete the neurons
void Network::deleteNeurons()
{
  if(created)
    for(unsigned int i = 0; i < hiddenUnits.size(); ++i) 
      delete hiddenUnits[i];
  created = false;
}

//! delete a network w/out deleting its neurons
void Network::releaseNeurons()
{
  for(unsigned int i = 0; i < hiddenUnits.size(); ++i)
    hiddenUnits[i] = NULL;
  delete this;
}

//! delete a network and neurons w/out deleting its weights
void Network::releaseWeights()
{
  for(unsigned int i = 0; i < hiddenUnits.size(); ++i)
    hiddenUnits[i]->releaseWeights();
  releaseNeurons();
  //delete this; already in releaseNeurons()
}

//NOTE
//! Set the Fitness 
void Network::setFitness(double fit, int obj)
{
  fitness->setFitness(fit, obj);
  /*
  ++trials;
  fitness += fit;
  */
}

/*
double Network::getFitness() 
{
  if(trials)
    return (double) fitness/trials; 
  else 
    return fitness;
}
*/



  
void Network::create() 
{
  //if(created) deleteNeurons();
  for (unsigned int i = 0; i < hiddenUnits.size(); ++i) {
    hiddenUnits[i] = new Neuron(geneSize);
    hiddenUnits[i]->create();
  }
  created = true;
}


bool Network::sizeEqual(Network &n)
{
  bool equal = true;
  
  if (!created || !n.created)
    equal = false;
  else if (hiddenUnits.size() != n.hiddenUnits.size())
    equal = false;
  else
    for(unsigned int i = 0; i < hiddenUnits.size(); ++i)
      if(hiddenUnits[i]->getSize() != n.hiddenUnits[i]->getSize()){
	 equal = false;
	 break;
      }

  return equal;
}



void Network::operator=(Network &n)
{
  
  if(!n.created){
    cerr << "assigning uncreated Network; Network::operator=" << endl;
    abort();
  }
  //check if nets are of the same type
  if( getType() != n.getType() ){
    cerr << "assigning networks of type " << n.getName()
         << " to type " << getName() << "; Network::operator=" << endl;
    abort();

    }
  activation = n.activation;
  trials  = n.trials;
  *fitness = *n.fitness;
  parent1 = n.parent1;
  parent2 = n.parent2;
  geneSize = n.geneSize;
  numInputs = n.numInputs;
  numOutputs = n.numOutputs;
  behavior = n.behavior;

  bias = n.bias;

  deleteNeurons();    //if created delete the neurons
  hiddenUnits.clear(); // resize(n.(int) hiddenUnits.size());
  //create();
  for(int i = 0; i < n.getNumNeurons(); ++i) {
    hiddenUnits.push_back( new Neuron(geneSize) );
    *hiddenUnits[i] = *n.hiddenUnits[i]; 
  }
  
  created = true;
}


bool Network::operator==(Network &n)
{
  bool equal = true;

  if(hiddenUnits.size() != n.hiddenUnits.size())
    equal = false;
  else
    for(unsigned int i = 0; i < hiddenUnits.size(); ++i){
      if(*hiddenUnits[i] != *n.hiddenUnits[i]){
	equal = false;
	break;
      }
    }
  return equal;
}

bool Network::operator!=(Network &n)
{
  if(*this == n) 
    return false;
  else 
    return true;
}

void Network::setNeuron(Neuron *n, int position)
{
  hiddenUnits[position] = n;
}

void Network::swapNeurons(int pos1, int pos2)
{
  Neuron *n = hiddenUnits[pos1];
  hiddenUnits[pos1] = hiddenUnits[pos2];
  hiddenUnits[pos2] = n;
}


Neuron* Network::getNeuron(int i) 
{ 
  if (i >= 0 && i < (int) hiddenUnits.size())
    return hiddenUnits[i];
  else {
    cerr << "Index out of bounds; Network::getNeuron" << endl;
    return NULL;
  }
}

inline
int Network::getParent(int p)
{
  if(p == 1)
    return parent1;
  else if(p == 2)
    return parent2;
  else{
    cerr <<  "Parent must be 1 or 2; Network::getParent" << endl;
    return -1;
  }
}



void Network::setNetwork(Network *n)
{
  parent1 = n->parent1;
  parent2 = n->parent2;
  fitness = n->fitness;
  trials = n->trials;
  for(unsigned int i = 0; i <  hiddenUnits.size(); ++i) 
    hiddenUnits[i] = n->hiddenUnits[i];

}

void Network::assignFitness(double fit, int obj)
{
  fitness->setFitness(fit, obj);
  for(unsigned int i = 0; i < hiddenUnits.size(); ++i){
      hiddenUnits[i]->assignFitness(fit, obj); 
  }
}

void Network::resetActivation()
{
  for(unsigned int i=0; i < activation.size();++i)
    activation[i] = 0.0;
}


//  void Network::perturb(Network *net)
//  {
//    for(int i=0;i<(int) hiddenUnits.size();++i){
//      net->setNeuron( hiddenUnits[i]->perturb(0.2), i);
//    }
//  }

//used by NetworkLevel
void Network::perturb(Network *net)
{
  for(unsigned int i=0 ; i < hiddenUnits.size();++i){
    hiddenUnits[i]->perturb(net->hiddenUnits[i], rndCauchy, 0.1);
  }
}

//---------------------------------------------------------------------
//! Same as above but called on self and returns new Network
Network* Network::perturb(double coeff)
{
  Network *n = this->clone();
  //*n = *this;
  for(unsigned int i = 0 ; i < hiddenUnits.size() ; ++i) 
    n->hiddenUnits[i] = hiddenUnits[i]->perturb(coeff);
  n->created = true;
  return n;
}

void Network::mutate(double mutRate)
{
  if(drand48() < mutRate)
    hiddenUnits[lrand48()%(int) hiddenUnits.size()]->mutate();
}
  
   
void Network::lesion(int i)
{ 
  if((i >= 0) && i < (int) hiddenUnits.size())
    hiddenUnits[i]->lesioned = true;
}

void Network::unlesion()
{
  for(unsigned int i = 0; i < hiddenUnits.size(); ++i)
    hiddenUnits[i]->lesioned = false;
}

//----------------------------------------------------------------------

//may eliminate these two or give it a better interface
void Network::addConnection(int locus)
{
  // if(locus < geneSize)
  for(unsigned int i=0; i < hiddenUnits.size(); ++i)
    hiddenUnits[i]->addConnection(locus);
}

//----------------------------------------------------------------------
// opposite of addConnection.
void Network::removeConnection(int locus)
{
  if(locus < geneSize)
    for(unsigned int i=0; i < hiddenUnits.size(); ++i)
      hiddenUnits[i]->removeConnection(locus);
}


//----------------------------------------------------------------------
//
void Network::saveXML(char *fname)
{
  ofstream  netfile(fname);

  if (!netfile) {
    cerr << "Error - cannot open" << fname << "(in Network::saveText)" << endl;
    exit(1);
  }
  cout << "Saving network to " << fname << endl;  
  
  netfile << "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>" << endl;
  netfile << "<!DOCTYPE NETWORK SYSTEM \"network.dtd\">" << endl;
  //netfile << "<NET_TYPE>" << type << "</NET_TYPE>" << endl;
  netfile << *this << endl;
  netfile.close();
}



bool NORMAL_MAPPING = true;
bool HIDDEN_TOGETHER = false;
//Create the network weight vector from the neurons
void Network::makeWtVector()
{
  unsigned int i;
  int j;

  weights.clear();
  
  if(NORMAL_MAPPING){
    for(i = 0; i < hiddenUnits.size(); ++i)
      for(j = 0; j < geneSize; ++j)
	weights.push_back( hiddenUnits[i]->getWeight(j) ); 
  }
  else if(HIDDEN_TOGETHER){
    //Put all inputs, recurrent, and output weights next to 
    // each other. First from one neuron, then the next...
    for(i = 0; i < hiddenUnits.size(); ++i)
      for(j = 0; j < numInputs; ++j)
	weights.push_back( hiddenUnits[i]->getWeight(j) );
    for(i = 0; i < hiddenUnits.size(); ++i)
      for(j = numInputs; j < geneSize; ++j)
	weights.push_back( hiddenUnits[i]->getWeight(j) );
    //  for(unsigned int i = 0; i < weights.size(); ++i)
    // cout << weights[i] << " ";
    //cout << endl;
  }
  else{ //INPUTS_TOGETHER
    for(j = 0; j < numInputs; ++j)
      for(i = 0; i < hiddenUnits.size(); ++i)
	weights.push_back( hiddenUnits[i]->getWeight(j) );
    for(j = numInputs; j < geneSize; ++j)   
      for(i = 0; i < hiddenUnits.size(); ++i)
      	weights.push_back( hiddenUnits[i]->getWeight(j) );
  }
}

//Map the network weight vector back to the 
//neuron weight vectors
void Network::mapBackToNeurons()
{
  unsigned int i;
  int j;
  
  if(NORMAL_MAPPING){
    for(i = 0; i < hiddenUnits.size(); ++i)
      for(j = 0; j < geneSize; ++j)
	hiddenUnits[i]->setWeight(j, weights[i*geneSize+j] ); 
  }
  else if(HIDDEN_TOGETHER){
    for(i = 0; i < hiddenUnits.size(); ++i)
      for(j = 0; j < numInputs; ++j)
	//hiddenUnits[i]->setWeight(j, weights[i*numInputs+j] );
	hiddenUnits[i]->setAllele(j, weights[i*numInputs+j] );
    
    int recurrent = geneSize - numInputs;
    int offset = numInputs * hiddenUnits.size();
    for(i = 0; i < hiddenUnits.size(); ++i)
      for(j = 0; j < recurrent; ++j){
	//      hiddenUnits[i]->setWeight(numInputs+j, weights[i*recurrent+j+offset] );
	hiddenUnits[i]->setAllele(numInputs+j, weights[i*recurrent+j+offset] );
      }
  }
  else { //INPUTS TOGETHER
    for(j = 0; j < numInputs; ++j)    
      for(i = 0; i < hiddenUnits.size(); ++i)
	hiddenUnits[i]->setAllele(j, weights[j*hiddenUnits.size()+i] );
    
    int recurrent = geneSize - numInputs;
    int offset = numInputs * hiddenUnits.size();
    for(j = 0; j < recurrent; ++j){   
      for(i = 0; i < hiddenUnits.size(); ++i)
      	hiddenUnits[i]->setAllele(numInputs+j, weights[j*hiddenUnits.size()+i+offset] );
    }
  }
  // for(unsigned int i = 0; i < weights.size(); ++i)
  // cout << weights[i] << " ";
  //cout << endl;
}

