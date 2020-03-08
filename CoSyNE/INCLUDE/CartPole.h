/*! \file CartPole.h
 */

#ifndef _CARTPOLE_H_
#define _CARTPOLE_H_

#include <vector>
#include <deque>
using std::deque;
#include "Environment.h"

struct cartArgs {
  bool markov;
  int numpoles;
  double polelength;
  double poleangle;
  double noise;
  double trajnoise;
  bool gruau;
  bool model;
  char modelfile[100];
  double gain;
  bool no_x;
  bool incremental;
};

class Network;
//! Pole Balancing domain
/*! 
    Implements the pole balancing dynamics using
    the Runge-Kutta 4th-order integration method.
    Can be instantiated with one or two poles.
*/
class CartPole : public Environment {
public:
  CartPole(struct cartArgs *);
  virtual ~CartPole(){}
  virtual bool simplifyTask();  
  virtual bool nextTask();
  double evalNetDump(Network *net, FILE *fptr);
  double generalizationTest(Network *net);
  void echoParams();
  double state[6]; //<! \todo  make private again!
  void performAction(const std::vector<double> &output); //<! \todo  make private again!
  void resetState(); //<! \todo  make private again!
  cartArgs *arguments;
  void setGruau (bool g){ gruau = g;}
  bool getGruau (){ return gruau;}
  void setNoise (double n){ noise = n;}
  double getNoise (){ return noise;}

protected:
  virtual void setupInput(std::vector<double> &input);
  void init();


private:
  int numPoles;
  bool initialized;
  bool reset;
  bool markov; // markov (full state) or non-markov (no-velocities).
  bool gruau;
  deque< double* > stateQ;
  double longPoleAngle;
  bool useModel;
  double noise;
  double trajNoise;
  double gain; // NEW_EXP
  bool noX;
  double dydx[6];

  virtual double evalNet(Network *);

  void step(double action, double *state, double *derivs);
  void rk4(double f, double y[], double dydx[], double yout[]);
  bool outsideBounds();

};

#endif

