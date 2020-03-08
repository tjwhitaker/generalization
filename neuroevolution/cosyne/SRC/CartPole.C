
#include <cstdlib>
#include <cstdio>
#include <iostream>
using std::cout;
using std::endl;

#include <vector>
using std::vector;

#include <deque>
using std::deque;

#include <math.h>
#include "Fitness.h"
#include "Network.h" 
#include "signal-handler.h"
#include "CartPole.h"

#define sgn(x)                  ((x >= 0) ? 1 : -1)

//////////////////////////////////////////////////////////////////////
//
// Double Pole physics
//
//////////////////////////////////////////////////////////////////////

static const double TRACK_SIZE  = 2.4;
static const double MUP         = 0.000002;
static const double MUC         = 0.0005;
static const double GRAVITY     = -9.8;
static const double MASSCART    = 1.0;
static const double MASSPOLE_1  = 0.1;
double MASSPOLE_2        = 0.01;
static const double LENGTH_1    = 0.5;	 // actually half the pole's length 
double LENGTH_2          = 0.05;
static const double FORCE_MAG   = 10.0;  //magnitude of max force
static const double FORCE_MAG2   = 20.0;  //magnitude of max force times 2
static const double TAU         = 0.01;	  //seconds between state updates 

#define BIAS 1
#define one_degree 0.0174532	/* 2pi/360 */
#define six_degrees 0.1047192
#define twelve_degrees 0.2094384
#define fifteen_degrees 0.2617993
#define thirty_six_degrees 0.628329
#define degrees64    1.2566580
#define fifty_degrees 0.87266

int BANG_BANG = false;

int UNIFORM_NOISE = 0; 
#define GAUSSIAN_NOISE 0

//////////////////////////////////////////////////////////////////////
double rndGauss(double a, double d);


CartPole::CartPole(struct cartArgs *args)
  : numPoles(args->numpoles), 
    initialized(false),
    reset(false),
    markov(args->markov),
    stateQ(),
    longPoleAngle(args->poleangle * one_degree),
    useModel(args->model),
    noX(args->no_x), //should do these init in a func so that defaults can be set here
    gruau(args->gruau),
    arguments(args)
  
{  
  name = "CartPole";
  LENGTH_2 = args->polelength/2;
  MASSPOLE_2 = args->polelength/10;
  noise = args->noise/50;
  trajNoise = args->trajnoise/100.0;  //!< make a percent later.
  incremental = args->incremental; //incremental: member of Environment
  init();
}


//#define _OLD_INPUT_SCHEME
void CartPole::init()
{
  switch (numPoles){
  case 1:
    if(!markov) {         //**********************
      inputDimension = 2; //3 //NO BIAS
                          //**********************
      if(noX) --inputDimension; //no cart position info (x)
    }
    else inputDimension = 4;
    break;
  case 2:
    BANG_BANG = false;
    if(!markov) {
#ifdef _OLD_INPUT_SCHEME
      inputDimension = 6;
#else
      inputDimension = 3;
#endif   
    }
    else inputDimension = 6;
    break;
  };
#ifdef BIAS
  inputDimension++;
#endif   

  //do this once nox works for all configs
  //  if(noX)
  //  --inputDimension; //no cart position info (x)

  outputDimension = 1;

  if(gruau)  goalFitness = 1001; 
  else goalFitness = 100000; 

  initialized = true;
  echoParams();
}

void CartPole::echoParams(){
  cout << endl << "#Cart-pole environment settings:";
  cout << endl << "#-------------------------------" << endl;
  cout << "#Number of poles            : " << numPoles << endl;
  cout << "#Length of short pole       : " << LENGTH_2 * 2 << " meters" << endl;
  cout << "#Initial angle of long pole : " << longPoleAngle/one_degree 
       << " degrees" << endl;
  cout << "#Number of inputs           : " << inputDimension << endl;	 
  cout << "#Goal fitness               : " << goalFitness << endl;
  if(markov)
    cout << "#Markov -- full state information." << endl;
  else
    cout << "#Non-Markov -- no velocity information." << endl;
  if(noX)
    cout << "#No Cart position (x) information." << endl;
  if(noise)
    cout << "#Percent sensor noise    : " << noise * 50 << endl;
  if(trajNoise)
    cout << "#Percent trajectory noise    : " << trajNoise * 100.0 << endl;
  if(gruau)
    cout << "#Using Gruau fitness." << endl;
  if(incremental) 
    cout << "#Evolving incrementally" << endl;
  if(BANG_BANG)
    cout << "#BANG BANG control" << endl;
}

  
void CartPole::resetState()
{
  int i,j;

  dydx[0] = dydx[1] = dydx[2] = dydx[3] = dydx[4] = dydx[5] = 0.0;
  state[0] = state[1] = state[3] = state[4] = state[5] = 0;
  state[2] = longPoleAngle;
  reset = true;
}  


double CartPole::evalNet(Network *net)
{
  int i,steps = 0;
  vector<double> input(inputDimension), output(outputDimension);
  vector<double> modelInput(6), modelOutput(6);
  double *tmp;
  deque<double> delayQ(0);
  double f1,f2 = 0;
  deque<double> x(0), xDot(0), theta(0), thetaDot(0);
  double fit;

  if(!initialized) init();
  
  if(!reset) resetState();

  while (steps++ < goalFitness) {
    setupInput(input);


    net->activate(input, output);
    if(gruau){
      x.push_front(input[0]);
      xDot.push_front(input[1]);
      theta.push_front(input[2]);
      thetaDot.push_front(input[3]);
      if(x.size() > 100){
	x.pop_back();
	xDot.pop_back();
	theta.pop_back();
	thetaDot.pop_back();
      }
    }
    performAction(output);
    
    /* cout << input[0] << " "
	 << input[1] << " "
	 << input[2] << " "
	 << input[3] << " "
	 << output[0] << endl; */
    if (outsideBounds())	// if failure
      break;			// stop it now
  }
  
  
  reset = false;

  if(gruau){
    f1 =  (double) steps/1000;
    if(steps < 100)
      f2 = 0;
    else{
      for(int i = 0; i < 100; ++i)
	f2 += fabs(x[i]) + fabs(xDot[i]) + fabs(theta[i]) + fabs(thetaDot[i]);
      f2 = 0.75/f2;
    }
    fit = 0.1 * f1 + 0.9 * f2;
  }
  else
    fit = (double) steps;
  
  return fit;  
}


double CartPole::evalNetDump(Network *net, FILE *fptr)
{
}



#define one_over_256  0.0390625
void CartPole::step(double action, double *st, double *derivs)
{
  double force;
  double costheta_1, costheta_2;
  double sintheta_1, sintheta_2;
  double gsintheta_1,gsintheta_2; 
  double temp_1,temp_2;
  double ml_1, ml_2;
  double fi_1,fi_2=0.0;
  double mi_1,mi_2=0.0;

  // action += (drand48() - 0.5) * trajNoise;
  //using tanh net output!!
  if(BANG_BANG && numPoles == 1){
    if(action > 0) force = FORCE_MAG;
    else force = -FORCE_MAG;
  }
  else
    //  force =  action * FORCE_MAG;
    force =  (action - 0.5) * FORCE_MAG2;
  
  if(isnan(force)) //check for nan created by huge weights
     force = FORCE_MAG;

  if((force >= 0) && (force < one_over_256))
    force = one_over_256;
  if((force < 0) && (force > -one_over_256))
    force = -one_over_256;
  

  costheta_1 = cos(st[2]);
  sintheta_1 = sin(st[2]);
  gsintheta_1 = GRAVITY * sintheta_1;
  ml_1 = LENGTH_1 * MASSPOLE_1;   
  temp_1 = MUP * st[3] / ml_1;
  fi_1 = (ml_1 * st[3] * st[3] * sintheta_1) +
    (0.75 * MASSPOLE_1 * costheta_1 * (temp_1 + gsintheta_1));
  mi_1 = MASSPOLE_1 * (1 - (0.75 * costheta_1 * costheta_1));

  if(numPoles > 1){
    costheta_2 = cos(st[4]);
    sintheta_2 = sin(st[4]);
    gsintheta_2 = GRAVITY * sintheta_2;
    ml_2 = LENGTH_2 * MASSPOLE_2;
    temp_2 = MUP * st[5] / ml_2;
    fi_2 = (ml_2 * st[5] * st[5] * sintheta_2) +
      (0.75 * MASSPOLE_2 * costheta_2 * (temp_2 + gsintheta_2));
    mi_2 = MASSPOLE_2 * (1 - (0.75 * costheta_2 * costheta_2));
  }
  
  derivs[1] = (force - MUC * sgn(st[1]) + fi_1 + fi_2)
	/ (mi_1 + mi_2 + MASSCART);
  
  derivs[3] = -0.75 * (derivs[1] * costheta_1 + gsintheta_1 + temp_1)
    / LENGTH_1;
  if(numPoles > 1)
    derivs[5] = -0.75 * (derivs[1] * costheta_2 + gsintheta_2 + temp_2)
      / LENGTH_2;

}

void CartPole::rk4(double f, double y[], double dydx[], double yout[])
{
  
  int i;
  
  double hh,h6,dym[6],dyt[6],yt[6];
  int vars = 3;
  
  if(numPoles > 1)
    vars = 5;
       
  hh=TAU*0.5;
  h6=TAU/6.0;
  for (i=0;i<=vars;i++) yt[i]=y[i]+hh*dydx[i];
  step(f,yt,dyt);
  dyt[0] = yt[1];
  dyt[2] = yt[3];
  dyt[4] = yt[5];
  for (i=0;i<=vars;i++) yt[i]=y[i]+hh*dyt[i];
  step(f,yt,dym);
  dym[0] = yt[1];
  dym[2] = yt[3];
  dym[4] = yt[5];
  for (i=0;i<=vars;i++) {
    yt[i]=y[i]+TAU*dym[i];
    dym[i] += dyt[i];
  }
  step(f,yt,dyt);
  dyt[0] = yt[1];
  dyt[2] = yt[3];
  dyt[4] = yt[5];
  for (i=0;i<=vars;i++)
    yout[i]=y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);
}
	

void CartPole::setupInput(vector<double> &input)
{
  switch(numPoles){
  case 1:
    if(markov){
	input[0] = state[0] / 2.4                + (drand48() * noise - (noise/2));
	input[1] = state[1] / 10.0               + (drand48() * noise - (noise/2));
	input[2] = state[2] / twelve_degrees     + (drand48() * noise - (noise/2));
	input[3] = state[3] / 5.0                + (drand48() * noise - (noise/2));
#ifdef BIAS
	input[4] = BIAS;
#endif
    }
    else{
      if(noX)
	input[0] = state[2] / twelve_degrees     + (drand48() * noise - (noise/2));
      else{
	input[0] = state[0] / 2.4                + (drand48() * noise - (noise/2));
	input[1] = state[2] / twelve_degrees     + (drand48() * noise - (noise/2));
      }
      //NO BIAS input[2] = BIAS;
    }
    break;
  case 2:
    if(markov){
      input[0] = state[0] / 2.4                + (drand48() * noise - (noise/2));
      input[1] = state[1] / 10.0               + (drand48() * noise - (noise/2));
      input[2] = state[2] / thirty_six_degrees + (drand48() * noise - (noise/2));
      input[3] = state[3] / 5.0                + (drand48() * noise - (noise/2));
      input[4] = state[4] / thirty_six_degrees + (drand48() * noise - (noise/2));
      input[5] = state[5] / 16.0               + (drand48() * noise - (noise/2));
#ifdef BIAS
      input[6] = BIAS;
#endif
    }
    else{
#ifdef _OLD_INPUT_SCHEME
      input[0] = state[0] / 2.4                + (drand48() * noise - (noise/2)); //4.8?
      input[1] = 0.0;
      input[2] = state[2] / thirty_six_degrees + (drand48() * noise - (noise/2));
      input[3] = 0.0;
      input[4] = state[4] / thirty_six_degrees + (drand48() * noise - (noise/2));
      input[5] = 0.0;
#ifdef BIAS
      input[6] = BIAS;
#endif
#else
      input[0] = state[0] / 2.4                + (drand48() * noise - (noise/2));
      input[1] = state[2] / 0.52               + (drand48() * noise - (noise/2));
      input[2] = state[4] / 0.52               + (drand48() * noise - (noise/2));
#ifdef BIAS
      input[3] = BIAS;
#endif      
#endif
    }
      break;
  };
}

#define RK4 1
#define EULER_TAU (TAU/8)
void CartPole::performAction(const vector<double> &output)
{ 
  
  int i;
  double tmpState[6];
  double force;

  force = output[0];
  /*random start state for long pole*/
  /*state[2]= drand48();   */
  
    
  /*--- Apply action to the simulated cart-pole ---*/

  if(RK4){      
    dydx[0] = state[1];
    dydx[2] = state[3];
    dydx[4] = state[5];
    step(force,state,dydx);
    rk4(force,state,dydx,state);
    for(i=0;i<6;++i)
      tmpState[i] = state[i];
    dydx[0] = state[1];
    dydx[2] = state[3];
    dydx[4] = state[5];
    step(force,state,dydx);
    rk4(force,state,dydx,state);
  }
  else{
    for(i=0;i<16;++i){
      step(output[0],state,dydx);
      state[0] += EULER_TAU * state[1];
      state[1] += EULER_TAU * dydx[1];
      state[2] += EULER_TAU * state[3];
      state[3] += EULER_TAU * dydx[3];
      state[4] += EULER_TAU * state[5];
      state[5] += EULER_TAU * dydx[5];
    }
  }
}


bool CartPole::outsideBounds()
{
  double failureAngle; 

  if(numPoles > 1){
    failureAngle = thirty_six_degrees; 
    return 
      fabs(state[0]) > TRACK_SIZE       || 
      fabs(state[2]) > failureAngle     ||
      fabs(state[4]) > failureAngle;  
  }
  else{
    failureAngle = twelve_degrees; 
    return 
      fabs(state[0]) > TRACK_SIZE       || 
      fabs(state[2]) > failureAngle;
  }
}


#define MIN_INC 0.0001
double POLE_INC = 0.01;//0.05;
double MASS_INC = 0.005;//0.01;

bool CartPole::nextTask()
{
#ifdef MODE
  MEM_LEVEL *= 2;
  cout << "MEM " << MEM_LEVEL << endl; 
  return false;
#else

  if(LENGTH_2 < 0.35){
    LENGTH_2 += POLE_INC;   /* LENGTH_2 * INCREASE;   */
    MASSPOLE_2 += MASS_INC; /* MASSPOLE_2 * INCREASE; */

    cout << "#Pole Length " << LENGTH_2 * 2 << " meters" << endl;
    return false;
  } 
  else
    return true;
#endif
  //noise += 0.1;
  //printf("#NOISE %2.4f\n", noise);
  //printf("#DONE \n");
  //  gInterrupt = true;  
}

bool CartPole::simplifyTask()
{
#ifndef MODE
  static int count = 0;
  bool val = true;

  if(POLE_INC > MIN_INC) 
    {
      POLE_INC = POLE_INC/2;
      MASS_INC = MASS_INC/2;
      LENGTH_2 -= POLE_INC;
      MASSPOLE_2 -= MASS_INC;
      cout << "#SIMPLIFY\n" << endl;
      cout << "#Pole Length " << LENGTH_2 * 2 << " meters" << endl;
    }
  else
    {
      cout << "#NO TASK CHANGE" << endl;
      if(++count > 2)
	val = false;
    }
  return val;
#endif
}



double rndGauss(double a, double d)
{
  static double t = 0;

  double v1,v2,r;

  if( t == 0 ){
    do{
      v1 = 2.0*drand48() - 1.0;
      v2 = 2.0*drand48() - 1.0;
      r = v1*v1+v2*v2;
    }  while (r >= 1);
    r = sqrt((-2.0*log(r))/r);
    t = v2*r;
    return  a+v1*r*d;
  }
  else {
    t = 0;
    return a+t*d;
  }
}

double CartPole::generalizationTest(Network *net)
{
  int i,j,k,m, success = 0, score;
  double testMax = 1000;
  double intervals[5] = { 0.05, 0.25, 0.5, 0.75, 0.95 };
  double tmp, tmpNoise;
 
  tmp = goalFitness;
  tmpNoise = noise;
  noise = 0;
  goalFitness = testMax;
  
  for(i = 0; i < 5; ++i)
    for(j = 0; j < 5; ++j)
      for(k = 0; k < 5; ++k)
	for(m = 0; m < 5; ++m){
	  reset = true; //make sure state is not reset to 'eval' initial state
	  state[0] = intervals[i] * 4.32 - 2.16;
	  state[1] = intervals[j] * 2.70 - 1.35;
	  state[2] = intervals[k] * 0.12566304 - 0.06283152;
	  /* 0.03141576; 3.6 degrees */
	  state[3] = intervals[m] * 0.30019504 - 0.15009752; /* 0.07504876;  8.6 degrees */
	  state[4] = 0.0;
	  state[5] = 0.0;
	  // printf("%f %f %f %f ", state[0], state[1], state[2], state[3]);	  
	  score = (int) evalNet(net);

	  //	  printf(" %d\n", score);
	  if(score >= testMax){
	    ++success;
	    printf(".");
	    fflush(stdout);
	  }
	}
  noise = tmpNoise;
  goalFitness = tmp;
  printf("Number of successfull trials : %d\n", success);
  return (double) success;
} 

