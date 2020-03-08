#ifndef _FITNESS_
#define _FITNESS_

/*! \file Fitness.h

*/

#include <vector>
using std::vector;

#include <iostream>
using std::endl;
using std::cout;
using std::cerr;


class Fitness {
  
public:
  static int numObjectives;  

  Fitness(int objectives)
    : fitnesses(objectives),
      sum(objectives)
  { reset(); }
  
  Fitness()
    : fitnesses(numObjectives),
      sum(numObjectives)
  {  reset(); }
  
  //~Fitness();
  
  
  void reset() {
    for(unsigned int i=0; i < fitnesses.size(); ++i){
      fitnesses[i].clear();
      sum[i] = 0;;
    }
  }
  
  bool operator>(Fitness &n)
  {
    bool result = true;
    for(unsigned int i=0; i < fitnesses.size(); ++i){
      if(fitnesses[i] < n.fitnesses[i]){
	result = false;
	break;
      }
      if(fitnesses[i] > n.fitnesses[i])
	break;
    }  
    return result;
  }    
  
  inline int getNumTrials(int obj = 0){
    return fitnesses[obj].size();
  }

  inline double getMinFitness(int obj = 0){   
    if(fitnesses[obj].size())
      return min; 
    else
      return 0;
  }
  inline double getMaxFitness(int obj = 0){   
    if(fitnesses[obj].size())
      return max; 
    else
      return 0;
  }
  inline double getAveFitness(int obj = 0){
    if(fitnesses[obj].size())
      return (double) sum[obj]/fitnesses[obj].size();
    else
      return 0;
  }
  
  void setFitness(double fit, int obj = 0){ 
    if(obj >= (int) fitnesses.size() ){
      cerr << "ERROR (Fitness::setFitness()): " << obj << endl;
      exit(0);
    }
    if(fitnesses[obj].empty() )
      min = max = fit;
    else if( fit > max )
      max = fit;
    else if( fit < min )
      min = fit;

    fitnesses[obj].push_back(fit);
    sum[obj] += fit;
  }

  
  int getNumObjectives() { return fitnesses.size(); }
 
  

private:
  vector< vector<double> > fitnesses;
  vector< double > sum;
  double min;
  double max;

};
/*
std::ostream& operator<<(std::ostream& os, Fitness &fit)
    {
      os << fit.getMaxFitness() << endl;

      return os;
    }
*/
//extern int Fitness::numObjectives;
#endif
