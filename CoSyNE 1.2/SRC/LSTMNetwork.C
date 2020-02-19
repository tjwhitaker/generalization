///////////////////////////////////////////////////////////////////
//
//  LSTMNetwork class
//
//  Long Short-Term memory network  
//  Copyright (C) 2007  Faustino J. Gomez
//
///////////////////////////////////////////////////////////////////


#include "LSTMNetwork.h"
#include "Neuron.h"
#include <iostream>
using std::cout; using std::endl;

LSTMNetwork::LSTMNetwork(int in, int hid, int out)
  : Network(in, hid, out),
    tmp(),
    state(hid)
{
  geneSize = 4*(in+hid)+out;
}

void LSTMNetwork::activate(std::vector<double> &input,
				      std::vector<double> &output)
{
  register int i,j;
  std::vector<double> netin(hiddenUnits.size());
  std::vector<double> netforget(hiddenUnits.size());
  std::vector<double> netcell(hiddenUnits.size());
  std::vector<double> netout(hiddenUnits.size());
  tmp = input;

  // evaluate hidden/output layer
  for (i=0;i<(int) hiddenUnits.size();++i){
    tmp.push_back(activation[i]);
  }
  for (i=0;i<(int) hiddenUnits.size();++i) {  /*for each hidden unit*/
      netin[i] = 0.0;
      netforget[i] = 0.0;
      netcell[i] = 0.0;
      netout[i] = 0.0;
      if(!hiddenUnits[i]->lesioned)
      {
        for (j=0;j<numInputs+(int) hiddenUnits.size();++j)
        {
        	netin[i] += hiddenUnits[i]->getWeight(j*4) * tmp[j];
        	netforget[i] += hiddenUnits[i]->getWeight(j*4 + 1) * tmp[j];
        	netcell[i] += hiddenUnits[i]->getWeight(j*4 + 2) * tmp[j];
        	netout[i] += hiddenUnits[i]->getWeight(j*4 + 3) * tmp[j];
        }
        netin[i] = sigmoid(netin[i] - 1.0);
        netforget[i] = sigmoid(netforget[i] + 3.0);
        //netcell[i] = sigmoid(netcell[i]);
        netout[i] = sigmoid(netout[i] - 2.0);
      }
  }
  for(i=0; i<(int) hiddenUnits.size(); i++) {
      state[i] = netcell[i]*netin[i] + /*netforget[i]*/state[i];
      if (i<3 ) activation[i] = state[i] * netout[i];
      else activation[i] = netcell[i];
  }

  for(i=0;i<numOutputs;++i)
    {
      output[i] = 0.0;
    for (j=0;j<(int) hiddenUnits.size();++j)
    {
      output[i] += activation[j] * hiddenUnits[j]->getWeight(4*numInputs + 4*hiddenUnits.size() + i);
    }
    if(0)//i == 0)
      output[i] = sigmoid( output[i] );
  }
}

inline
void LSTMNetwork::growNeuron(Neuron *n)
{
  n->addConnection(4*numInputs + 4*(int) hiddenUnits.size() - 4);
  n->addConnection(4*numInputs + 4*(int) hiddenUnits.size() - 4);
  n->addConnection(4*numInputs + 4*(int) hiddenUnits.size() - 4);
  n->addConnection(4*numInputs + 4*(int) hiddenUnits.size() - 4);
}

inline
void LSTMNetwork::shrinkNeuron(Neuron *n, int pos = -1)
{
  n->removeConnection(numInputs + pos);
}


void LSTMNetwork::addNeuron()
{
  activation.push_back(0.0);
  state.push_back(0.0);
  addConnection(4*numInputs+4*(int) hiddenUnits.size());
  addConnection(4*numInputs+4*(int) hiddenUnits.size());
  addConnection(4*numInputs+4*(int) hiddenUnits.size());
  addConnection(4*numInputs+4*(int) hiddenUnits.size());
  geneSize = 4*numInputs + 4*(int) hiddenUnits.size() + numOutputs + 4;
  hiddenUnits.push_back( new Neuron(geneSize) );
}

void LSTMNetwork::removeNeuron(int sp)
{
  if( hiddenUnits.size() > 1 ){
    activation.pop_back();
    removeConnection(numInputs+sp);
    delete hiddenUnits[sp];
    hiddenUnits.erase(hiddenUnits.begin()+sp);
    geneSize = 4*numInputs + 4*(int) hiddenUnits.size() + numOutputs;
  }
}

//reset state and call base class resetActivation
void LSTMNetwork::resetActivation()
{
  Network::resetActivation();
  for(unsigned int i=0; i < hiddenUnits.size();++i) {
    state[i] = 0.0;
  }
}
