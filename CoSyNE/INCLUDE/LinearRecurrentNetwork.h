//! Linear Recurrent Network (LRN)
/*!     Like \c FeedForwardNetwork (1 hidden layer) except that, like 
    \c FullyReccurentNetwork the neuron activation
    is fed back into the network, and no sigmoid on output
*/

#include "Network.h"

class LinearRecurrentNetwork : public Network {
public:
  LinearRecurrentNetwork(int in, int hid, int out);
  Network* newNetwork(int in, int hid, int out){ 
    return new LinearRecurrentNetwork(in, hid, out); }
  Network* clone(){ return new LinearRecurrentNetwork(numInputs, hiddenUnits.size(), numOutputs); }  
  void activate(std::vector<double> &input, std::vector<double> &output);
  void growNeuron( Neuron * );
  void shrinkNeuron( Neuron *, int );
  void addNeuron();
  void removeNeuron(int);
private:
  std::vector<double> tmp;
};
