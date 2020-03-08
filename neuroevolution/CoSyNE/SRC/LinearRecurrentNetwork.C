///////////////////////////////////////////////////////////////////
// 
//  LinearRecurrentNetwork class
//  
//  Simple Recurrent Network with linear output units
//  Copyright (C) 2007  Faustino J. Gomez
//
//////////////////////////////////////////////////////////////////

#include "LinearRecurrentNetwork.h"
#include "Neuron.h"
#include <iostream>
using std::cout; using std::endl;

LinearRecurrentNetwork::LinearRecurrentNetwork(int in, int hid, int out) 
  : Network(in, hid, out), 
    tmp()
{ 
  geneSize = in+hid+out; 
}

void LinearRecurrentNetwork::activate(std::vector<double> &input, 
				      std::vector<double> &output)
{ 
  register int i,j;

  tmp = input;

  // evaluate hidden/output layer 
  for (i=0;i<(int) hiddenUnits.size();++i){
    tmp.push_back(activation[i]);
    //cout << activation[i] << endl;
  }
  for (i=0;i<(int) hiddenUnits.size();++i) {  /*for each hidden unit*/
    activation[i] = 0.0;
    if(!hiddenUnits[i]->lesioned){
      for (j=0;j<numInputs+(int) hiddenUnits.size();++j)
	activation[i] += hiddenUnits[i]->getWeight(j) * tmp[j];
      activation[i] = sigmoid( activation[i] );
    }
  }
  for(i=0;i<numOutputs;++i){
    output[i] = 0.0;
    for (j=0;j<(int) hiddenUnits.size();++j){
      output[i] += activation[j] * hiddenUnits[j]->getWeight(numInputs + hiddenUnits.size() + i);
    }
  }
}

inline
void LinearRecurrentNetwork::growNeuron(Neuron *n)
{  
  n->addConnection(numInputs + (int) hiddenUnits.size() - 1);
}

inline
void LinearRecurrentNetwork::shrinkNeuron(Neuron *n, int pos = -1)
{
  n->removeConnection(numInputs + pos);
}


void LinearRecurrentNetwork::addNeuron()
{
  activation.push_back(0.0);
  addConnection(numInputs+(int) hiddenUnits.size() );
  geneSize = numInputs + (int) hiddenUnits.size() + numOutputs + 1;  
  hiddenUnits.push_back( new Neuron(geneSize) );
}

void LinearRecurrentNetwork::removeNeuron(int sp)
{
  if( hiddenUnits.size() > 1 ){
    activation.pop_back();
    removeConnection(numInputs+sp);
    delete hiddenUnits[sp];
    hiddenUnits.erase(hiddenUnits.begin()+sp);
    geneSize = numInputs + (int) hiddenUnits.size() + numOutputs;  
  }
}
