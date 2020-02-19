///////////////////////////////////////////////////////////////////////// 
//  FullyRecurrenNetwork class
//
//  Fully recurrent neural network 
//  Copyright (C) 2007  Faustino J. Gomez
//
///////////////////////////////////////////////////////////////////////// 

#include <cassert>
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include "FullyRecurrentNetwork.h"
#include "Neuron.h"

FullyRecurrentNetwork::FullyRecurrentNetwork(int in, int hid, int out) 
  : Network(in, hid, out), 
    relax(1),
    tmp()
{ 
   if(out > hid) {
    cerr << "Not enough hidden units! (" << getName() << ")" << endl;
    exit(5);
  }
  geneSize = in+hid; 
}

#define THRESH 0.0
#define THRESH2 0.0
#define SIGMA_PI 
void FullyRecurrentNetwork::activate(std::vector<double> &input, 
				     std::vector<double> &output)
{ 
  register int i,j,r;

    //evaluate hidden/output layer 
  for (r=0;r<relax;++r){  
    tmp = input;
    for (i=0;i<(int) hiddenUnits.size();++i)
      tmp.push_back( activation[i] );
    for (i=0;i<(int) hiddenUnits.size();++i) {  //for each hidden unit
      assert(geneSize = hiddenUnits[i]->getSize() );
      activation[i] = 0.0;
      if(!hiddenUnits[i]->lesioned){
	for (j=0;j<numInputs + (int) hiddenUnits.size();++j)
	  activation[i] += hiddenUnits[i]->getWeight(j) * tmp[j];
	activation[i] = sigmoid( activation[i] ); 
	//activation[i] = tanh( activation[i] ); 
      } 
    }
    for(i=0;i<numOutputs;++i) 
      output[i] = activation[i];
  }  
}


inline
void FullyRecurrentNetwork::growNeuron(Neuron *n)
{  
  n->addConnection( n->getSize() );
}

inline
void FullyRecurrentNetwork::shrinkNeuron(Neuron *n, int pos = -1)
{
  n->removeConnection(numInputs + pos);
}



void FullyRecurrentNetwork::addNeuron()
{
  activation.push_back(0.0);
  addConnection(numInputs + (int) hiddenUnits.size() );
  geneSize = numInputs + (int) hiddenUnits.size() + 1;  
  hiddenUnits.push_back( new Neuron(geneSize) );
}

void FullyRecurrentNetwork::removeNeuron(int sp)
{
  if( hiddenUnits.size() > 1 ){
    activation.pop_back();
    removeConnection(numInputs+sp);
    delete hiddenUnits[sp];
    hiddenUnits.erase(hiddenUnits.begin()+sp);
    geneSize = numInputs + (int) hiddenUnits.size();  
  }
}



/////////////////////////////////////////////////////////////////
// FullyRecurrentNetwork2
//

FullyRecurrentNetwork2::FullyRecurrentNetwork2(int in, int hid, int out) 
  : Network(in, hid, out), 
    relax(2),
    tmp()
{ 
  if(out > hid) {
    cerr << "Not enough hidden units!" << endl;
    exit(5);
  }
  geneSize = in+hid;
} 

void FullyRecurrentNetwork2::activate(std::vector<double> &input, 
				     std::vector<double> &output)
{
  int max = 20, steps = 0;
  std::vector<double> prevOut(numOutputs);
  double epsilon = 0.05;
  register int i,j;

  prevOut[0] = 1000;
  //evaluate hidden/output layer 
  //  for (r=0;r<relax;++r){  
  while((fabs(output[0] - prevOut[0]) > epsilon) &&
	steps++ < max)
    { 
      for(i=0;i<numOutputs;++i) 
	prevOut[i] = output[i];
      tmp = input;
      for (i=0;i<(int) hiddenUnits.size();++i)
	tmp.push_back( activation[i] );
      for (i=0;i<(int) hiddenUnits.size();++i) {  //for each hidden unit
	activation[i] = 0.0;
	if(!hiddenUnits[i]->lesioned){
	  for (j=0;j<numInputs + (int) hiddenUnits.size();++j)
	    activation[i] += hiddenUnits[i]->getWeight(j) * tmp[j];
	  //activation[i] = sigmoid( activation[i] ); 
	  activation[i] = tanh( activation[i] ); 
	} 
      }
      for(i=0;i<numOutputs;++i) 
	output[i] = activation[i];
      
    }  
  //  cout << steps << endl;
}

/*
void FullyRecurrentNetwork2::activate(std::vector<double> &input, 
				      std::vector<double> &output)
				   
{ 
  register int i,j,r=0;
  bool done = false;


  while (!done){
    tmp = input;
    done = true;
    for (i=0;i<(int) hiddenUnits.size();++i)
      tmp.push_back( activation[i] );
    for (i=0;i<(int) hiddenUnits.size();++i) {  // for each hidden unit
      if((int) hiddenUnits[i]->getWeight( geneSize-1 ) > r){	
	done = false;
	activation[i] = 0.0;
	if(!hiddenUnits[i]->lesioned){
	  for (j=0;j<numInputs+(int) hiddenUnits.size();++j) 
	    activation[i] += hiddenUnits[i]->getWeight(j) * tmp[j];
	  activation[i] = sigmoid( activation[i] ); 
	}
      }
    }
    ++r;
  }
  for(i=0;i<numOutputs;++i) 
    output[i] = activation[i];
}
*/


inline
void FullyRecurrentNetwork2::growNeuron(Neuron *n)
{  
  n->addConnection( n->getSize() );
}

inline
void FullyRecurrentNetwork2::shrinkNeuron(Neuron *n, int pos = -1)
{
  n->removeConnection(numInputs + pos);
}

void FullyRecurrentNetwork2::addNeuron()
{
  activation.push_back(0.0);
  addConnection(numInputs+(int) hiddenUnits.size() );
  geneSize = numInputs + (int) hiddenUnits.size() + 1;  
  hiddenUnits.push_back( new Neuron(geneSize) );
}

void FullyRecurrentNetwork2::removeNeuron(int sp)
{
  if( hiddenUnits.size() > 1 ){
    activation.pop_back();
    removeConnection(numInputs+sp);
    delete hiddenUnits[sp];
    hiddenUnits.erase(hiddenUnits.begin()+sp);
    geneSize = numInputs + (int) hiddenUnits.size();  
  }
}
