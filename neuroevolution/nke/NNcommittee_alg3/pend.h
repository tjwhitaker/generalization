/*****************************************************************************************************************************\
*								 Class: Double Pole Balancing Problem		                                                 *
*         see:                                                                                                               *
*          Dynamical systems -> Wieland (1991). "Evolving neural network controllers for unstable systems", IJCNN.            *
*          Runge-Kutta -> Garcia (1997). "Modelagem e simulacao", EDUSP.                                                     *
*          Program -> Gomes (2002): ESP3.0, C++-code for evolving neural networks to solve the double pole balancing problem *
*                                                                                                                            *
\*****************************************************************************************************************************/

#include "defs.h"
#include <cmath>
#include <cstdlib>

#define size_st 6 // number of states
#define g 9.80665 // acceleration due to gravity

class doublePole {
private:
  double st[size_st];  // state: [x , dx, teta1, dteta1, teta2, dteta2]
  double dst[size_st]; // der. of state: [dx , ddx, dteta1, ddteta1, dteta2, ddteta2]
  double mi_c;         // coefficient of friction of the cart on the track
  double m_c;          // mass of the cart
  double mi_p[2];      // coefficient of friction of the poles
  double m_p[2];       // mass of the poles
  double l_p[2];       // half length of the poles
  double dt;           // integration step
  void rk4(double F, double *state, double *dstate);
  void dynSist(double F, double *state, double *dstate);

public:
  doublePole(double *st_init);
  ~doublePole(void);
  void step(double F, double *st_return);
  void print(void);
};

/******************************************************************************\
*								Constructor										*
\******************************************************************************/
doublePole::doublePole(double *st_init) {
  int i;

  // Parameters of the cart and poles
  mi_c = 0.0005;
  m_c = 1;
  mi_p[0] = 0.000002;
  mi_p[1] = 0.000002;
  m_p[0] = 0.1;
  m_p[1] = 0.01;
  l_p[0] = 0.5;
  l_p[1] = 0.05;

  // Integration step
  dt = 0.01;

  // States Initialization
  for (i = 0; i < size_st; i++)
    st[i] = st_init[i];
  dst[0] = st[1];
  dst[2] = st[3];
  dst[4] = st[5];
  dst[1] = dst[3] = dst[5] = 0.0;
}

/******************************************************************************\
*								 Destructor										*
\******************************************************************************/
doublePole::~doublePole(void) {
}

/******************************************************************************\
*						Dynamical System Equations                             *
\******************************************************************************/
void doublePole::dynSist(double F, double *state, double *dstate) {
  int i;
  double cos_theta[2], sin_theta[2];
  double m_p_eff[2]; // effective mass of the poles
  double F_p_eff[2]; // effective force from the poles on the cart

  // Cos and Sin of Theta
  for (i = 0; i < 2; i++) {
    cos_theta[i] = cos(state[2 * i + 2]);
    sin_theta[i] = sin(state[2 * i + 2]);
  }

  // Effective mass and effective force
  for (i = 0; i < 2; i++) {
    m_p_eff[i] = m_p[i] * (1.0 - 0.75 * pow(cos_theta[i], 2.0));
    F_p_eff[i] = m_p[i] * l_p[i] * pow(state[2 * i + 3], 2.0) * sin_theta[i] + 0.75 * m_p[i] * cos_theta[i] * ((mi_p[i] * state[2 * i + 3]) / (m_p[i] * l_p[i]) + g * sin_theta[i]);
  }

  // Dynamical system equations
  dstate[1] = (F - mi_c * sign(state[1]) + (F_p_eff[0] + F_p_eff[1])) / (m_c + m_p_eff[0] + m_p_eff[1]);
  for (i = 0; i < 2; i++)
    dstate[2 * i + 3] = -0.75 * (dstate[1] * cos_theta[i] + g * sin_theta[i] + (mi_p[i] * state[2 * i + 3]) / (m_p[i] * l_p[i])) / l_p[i];
}

/******************************************************************************\
*								4th Order Runge-Kutta  Method   			  *
\******************************************************************************/
void doublePole::rk4(double F, double *state, double *dstate) {
  int i;
  double u1[size_st], u2[size_st], u3[size_st], u4[size_st], state_temp[size_st], dstate_temp[size_st];

  // u1
  for (i = 0; i < size_st; i++)
    u1[i] = dt * dstate[i];

  // u2
  for (i = 0; i < size_st; i++)
    state_temp[i] = state[i] + (u1[i] / 2.0);
  dstate_temp[0] = state_temp[1];
  dstate_temp[2] = state_temp[3];
  dstate_temp[4] = state_temp[5];
  dynSist(F, state_temp, dstate_temp);
  for (i = 0; i < size_st; i++)
    u2[i] = dt * dstate_temp[i];

  // u3
  for (i = 0; i < size_st; i++)
    state_temp[i] = state[i] + (u2[i] / 2.0);
  dstate_temp[0] = state_temp[1];
  dstate_temp[2] = state_temp[3];
  dstate_temp[4] = state_temp[5];
  dynSist(F, state_temp, dstate_temp);
  for (i = 0; i < size_st; i++)
    u3[i] = dt * dstate_temp[i];

  // u4
  for (i = 0; i < size_st; i++)
    state_temp[i] = state[i] + u3[i];
  dstate_temp[0] = state_temp[1];
  dstate_temp[2] = state_temp[3];
  dstate_temp[4] = state_temp[5];
  dynSist(F, state_temp, dstate_temp);
  for (i = 0; i < size_st; i++)
    u4[i] = dt * dstate_temp[i];

  // final result
  for (i = 0; i < size_st; i++)
    state[i] = state[i] + (u1[i] + 2.0 * u2[i] + 2.0 * u3[i] + u4[i]) / 6.0;
  dstate[0] = state[1];
  dstate[2] = state[3];
  dstate[4] = state[5];
}

/******************************************************************************\
*								Simulation of one step of the dynamical system *
\******************************************************************************/
void doublePole::step(double F, double *st_return) {
  int i;
  double state[size_st], dstate[size_st];

  for (i = 0; i < size_st; i++) {
    state[i] = st[i];
    dstate[i] = dst[i];
  }

  dynSist(F, state, dstate);
  rk4(F, state, dstate);

  for (i = 0; i < size_st; i++) {
    dst[i] = dstate[i];
    st[i] = state[i];
    st_return[i] = st[i];
  }
}

/******************************************************************************\
*								Print Double Pole Information				  *
\******************************************************************************/
void doublePole::print(void) {

  cout << "Double Pole" << endl;
  cout << " States: " << endl;
  cout << "  Cart:" << endl;
  cout << "   Position (x): " << st[0] << endl;
  cout << "   Velocity (dx): " << st[1] << endl;
  cout << "   Acceleration (ddx): " << dst[1] << endl;
  cout << "  Pole 1:" << endl;
  cout << "   Position (theta_1): " << st[2] << endl;
  cout << "   Velocity (dtheta_1): " << st[3] << endl;
  cout << "   Acceleration (ddtheta_1): " << dst[3] << endl;
  cout << "  Pole 2:" << endl;
  cout << "   Position (theta_2): " << st[4] << endl;
  cout << "   Velocity (dtheta_2): " << st[5] << endl;
  cout << "   Acceleration (ddtheta_2): " << dst[5] << endl;
}
