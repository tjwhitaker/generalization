//! Single Layer FeedForward network.
/*! This \c Network type consists of a single layer
    of neurons w. Each neuron receives input from the
    environment \e and from the other neurons.  That is,
    each neuron's activation is fed back into the network.
    These networks have internal state and can be used in
    non-Markov task--task that require short-term memory.
*/

#ifndef _SINGLE_LAYER_FF
#define _SINGLE_LAYER_FF

#include "Network.h"

class SingleLayerFeedForwardNetwork : public  Network {
public:
  SingleLayerFeedForwardNetwork(int in, int hid, int out);
  Network* newNetwork(int in, int hid, int out){ 
    return new SingleLayerFeedForwardNetwork(in, hid, out); }  
  Network* clone(){ return new SingleLayerFeedForwardNetwork(numInputs, hiddenUnits.size(), numOutputs); }  

  void activate(std::vector<double> &, 
		std::vector<double> &);
  void growNeuron( Neuron * );
  void shrinkNeuron( Neuron *, int );
  void addNeuron();
  void removeNeuron(int);
  inline int getMinUnits() { return numOutputs; }
  
};

#endif
