#include <cstdlib>
#include <cmath>
#include "ne-utils.h"

double rndCauchy(double coeff) 
{
  double u = 0.5, Cauchy_cut = 10.0;
 
  while (u == 0.5) 
    u = drand48();
  u = coeff * tan(u * PI);
  if(fabs(u) > Cauchy_cut)
    return rndCauchy(coeff);
  else
    return u;
}
