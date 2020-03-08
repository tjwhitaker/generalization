///////////////////////////////////////////////////////////////////
// SingleLayerFeedForwardNetwork
//

#include <cassert>
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include "SingleLayerFeedForwardNetwork.h"
#include "Neuron.h"

SingleLayerFeedForwardNetwork::SingleLayerFeedForwardNetwork(int in, int hid, int out) 
  : Network(in, hid, out)
{ 
   if(out != hid) {
     cerr << "Net should contain " << out << " neurons! (" << getName() << ")" << endl;
    exit(5);
  }
  geneSize = in; 
}

void SingleLayerFeedForwardNetwork::activate(std::vector<double> &input, 
				     std::vector<double> &output)
{ 
  register int i,j;

    //evaluate hidden/output layer 
  for (i=0;i<(int) hiddenUnits.size();++i) {  //for each hidden unit
    assert(geneSize = hiddenUnits[i]->getSize() );
    activation[i] = 0.0;
    if(!hiddenUnits[i]->lesioned){
      for (j=0;j<numInputs;++j)
	activation[i] += hiddenUnits[i]->getWeight(j) * input[j];
      activation[i] = sigmoid( activation[i] ); 
    } 
  }
  for(i=0;i<numOutputs;++i) 
    output[i] = activation[i];
}  



inline
void SingleLayerFeedForwardNetwork::growNeuron(Neuron *n)
{  
  //n->addConnection( n->getSize() );
}

inline
void SingleLayerFeedForwardNetwork::shrinkNeuron(Neuron *n, int pos = -1)
{
  //n->removeConnection(numInputs + pos);
}



void SingleLayerFeedForwardNetwork::addNeuron()
{
  //activation.push_back(0.0);
  //addConnection(numInputs + (int) hiddenUnits.size() );
  //  geneSize = numInputs + (int) hiddenUnits.size() + 1;  
  hiddenUnits.push_back( new Neuron(geneSize) );
}

void SingleLayerFeedForwardNetwork::removeNeuron(int sp)
{
  if( hiddenUnits.size() > 1 ){
    activation.pop_back();
    removeConnection(numInputs+sp);
    delete hiddenUnits[sp];
    hiddenUnits.erase(hiddenUnits.begin()+sp);
    geneSize = numInputs + (int) hiddenUnits.size();  
  }
}



