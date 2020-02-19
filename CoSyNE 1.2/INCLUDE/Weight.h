#ifndef WEIGHT_H
#define WEIGHT_H

#include "Fitness.h"
#include "Neural.h"

class Weight : public Neural {
public:
  Weight(double _w = 0.0)
    : Neural(),
      w(_w),
      trials(0),
      moved(false),
      p(0)
  {}
  virtual ~Weight() {};
  double w;
  int trials;
  bool moved;
  double p;

  virtual void resetFitness() { fitness->reset(); }
  virtual void create() {}
  inline void setFitness (double fit, int obj = 0) { fitness->setFitness(fit, obj); }
};

#endif
