//! LSTM Network (LRN)


#include "Network.h"

class LSTMNetwork : public Network {
public:
  LSTMNetwork(int in, int hid, int out);
  Network* newNetwork(int in, int hid, int out){
    return new LSTMNetwork(in, hid, out); }
  Network* clone(){ return new LSTMNetwork(numInputs, hiddenUnits.size(), numOutputs); }
  void activate(std::vector<double> &input, std::vector<double> &output);
  void growNeuron( Neuron * );
  void shrinkNeuron( Neuron *, int );
  void addNeuron();
  void removeNeuron(int);
  virtual void resetActivation();

  std::vector<double> state;
private:
  std::vector<double> tmp;
  
};
