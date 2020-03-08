/////////////////////////////////////////////////////////////////////////
//  Cooperative Synapse NeuroEvolution (CoSyNE) 1.1Beta
//  Copyright (C) 2007  Faustino J. Gomez
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//////////////////////////////////////////////////////////////////////////

#include <unistd.h>
#include <argp.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>
using std::cout;
using std::endl;
#include "signal-handler.h"

#ifdef _CARTPOLE
#include "CartPole.h"
#endif

#ifdef _BLOCK
#include "BlockPacker.h"
#endif

#ifdef _MAZE89
#include "Maze89.h"
#endif

#include "Networks.h"
#include "CoSyNE.h"
#include "TestNets.h"

extern int NUM_LOCI;
static int maxGenerations;
static int maxEvaluations;

void reseed(int val);

#define OPT_NL 1
#define OPT_STAG 2
#define OPT_GROW 3
#define OPT_TEST 4
#define OPT_VERB 5
#define OPT_SEEDNET 6
#define OPT_SAVE 7
#define OPT_PERC 8
#define OPT_LESION 9
#define OPT_EVALS 10
#define OPT_INIT_SP 11
#define OPT_LOCI 12
#define OPT_NEUTRAL 13
#define OPT_WTRANGE 14
#define OPT_PROP 15
#define OPT_SHIFT 16
#define OPT_CROSS 17
#define OPT_FILE 18
#define OPT_LIST 19
#define OPT_EVALS_MAX 20
#define OPT_SAME_SEED 21
#define NUM_COSYNE_OPTS 21

bool TEST;
bool LESION;
char *NETFILE;

////////////////////////////////////////////////////////////////////
// Parse arguments

static struct cosyneArgs cosyneargs;

const char *argp_program_version = "CoSyNE 1.0";
const char *argp_program_bud_address = "<inaki@cs/utexas/edu>";

static char doc[] =
    "Enforced Sub Populations neuroevolution system.";
static char args_doc[] = "ARG1";

static struct argp_option options[] = {
    {"file", 'f', "file", 0, "Input file"},
    {"numpops", 'z', "Z", 0, "Number of subpopulations"},
    {"popsize", 'n', "N", 0, "Size of subpopulations"},
    {"nettype", 't', "FF,SRN,FR", 0, "Type of network"},
    {"seed", 's', "S", 0, "Random seed"},
    {"mutation", 'm', "M", 0, "Mutation rate"},
    {"generations", 'g', "G", 0, "Max number of generations"},
    {"netlevel", OPT_NL, "Size", 0, "Network level evolution"},
    {"dar", OPT_GROW, 0, 0, "Disable adding/removing of subpops"},
    {"stag", OPT_STAG, "S", 0, "Stagnation criteria"},
    {"test", OPT_TEST, "netfile", 0, "File of network to be tested"},
    {"verbose", OPT_VERB, 0, 0, "Print individual fitnesses"},
    {"seednet", OPT_SEEDNET, "netfile", 0, "Evolve around a network"},
    {"save", OPT_SAVE, 0, 0, "Save networks?"},
    {"perc", OPT_PERC, 0, 0, "Send Netlevel neurons back to ESP?"},
    {"lesion", OPT_LESION, 0, 0, "Lesion test"},
    {"nevals", OPT_EVALS, "N", 0, "Number of evals per neuron"},
    {"initsp", OPT_INIT_SP, "N", 0, "Initial subpop size"},
    {"loci", OPT_LOCI, "L", OPTION_HIDDEN, "Loci per weight"},
    {"neutral", OPT_NEUTRAL, "M", 0, "Neutral mutation rate"},
    {"proportional", OPT_PROP, 0, OPTION_HIDDEN, ""},
    {"shift", OPT_SHIFT, "double", OPTION_HIDDEN, ""},
    {"wtrange", OPT_WTRANGE, "W", OPTION_HIDDEN, "Allele value range"},
    {"crossover", OPT_CROSS, "int", 0, "Crossover type: 1=1p, 2=SBX"},
    {"list", OPT_LIST, 0, 0, "List network types"},
    {"evaluations", OPT_EVALS_MAX, "int", 0, "Max number of network evaluations"},
    {"ss", OPT_SAME_SEED, 0, OPTION_HIDDEN, "same seed for all evals"},
    {0}};

static error_t
parse_opt(int key, char *arg, struct argp_state *state)
{
  //struct cosyneargs *arguments = (struct cosyneargs *) state->input;

  switch (key)
  {
  case 'f':
    cosyneargs.paramfile = arg;
    break;
  case 'g':
    maxGenerations = atoi(arg);
    break;
  case 'z':
    if (atoi(arg) == 0)
    { //if -z is 0 chose random num of subpops.
      reseed(time(NULL));
      cosyneargs.numpops = (lrand48() % 9) + 1;
    }
    else
      cosyneargs.numpops = atoi(arg);
    break;
  case 'n':
    cosyneargs.popsize = atoi(arg);
    break;
  case 't':
    cosyneargs.net = genNet(intToNetEnum(atoi(arg)), 0, 0, 0);
    cosyneargs.nettype = atoi(arg); // change this
    break;
  case 's':
    cosyneargs.seed = atoi(arg);
    break;
  case 'm':
    cosyneargs.mutation = atof(arg);
    break;
  case OPT_NL:
    cosyneargs.netlevel = true;
    cosyneargs.netlevelsize = atoi(arg);
    break;
  case OPT_STAG:
    cosyneargs.stagnation = atoi(arg);
    ;
    break;
  case OPT_GROW:
    cosyneargs.grownets = false;
    break;
  case OPT_TEST:
    TEST = true;
    NETFILE = arg;
    //      printf("%s\n", NETFILE);
    break;
  case OPT_VERB:
    cosyneargs.verbose = true;
    break;
  case OPT_SEEDNET:
    cosyneargs.seednet = true;
    NETFILE = arg;
    break;
  case OPT_SAVE:
    cosyneargs.savenets = true;
    break;
  case OPT_PERC:
    cosyneargs.perc = true;
    break;
  case OPT_LESION:
    LESION = true;
    break;
  case OPT_EVALS:
    cosyneargs.numevals = atoi(arg);
    break;
  case OPT_INIT_SP:
    cosyneargs.init_popsize = atoi(arg);
    break;
  case OPT_LOCI:
    NUM_LOCI = atoi(arg);
    break;
  case OPT_NEUTRAL:
    cosyneargs.neutral = atof(arg);
    break;
  case OPT_PROP:
    cosyneargs.proportional = true;
    break;
  case OPT_SHIFT:
    cosyneargs.shift = atof(arg);
    break;
  case OPT_WTRANGE:
    cosyneargs.wtrange = atof(arg);
    break;
  case OPT_CROSS:
    cosyneargs.crossover_type = atoi(arg);
    break;
  case OPT_LIST:
    listNetTypes();
    exit(0);
    break;
  case OPT_EVALS_MAX:
    maxEvaluations = atoi(arg);
    break;
  case OPT_SAME_SEED:
    cosyneargs.same_seed = true;
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}
/////////////////////////////////////////////////////////////////

#ifdef _CARTPOLE
#define OPT_NM (NUM_COSYNE_OPTS + 1)
#define OPT_NPOLES (NUM_COSYNE_OPTS + 2)
#define OPT_PL (NUM_COSYNE_OPTS + 3)
#define OPT_NOISE (NUM_COSYNE_OPTS + 4)
#define OPT_ANGLE (NUM_COSYNE_OPTS + 5)
#define OPT_GRUAU (NUM_COSYNE_OPTS + 6)
#define OPT_MODEL (NUM_COSYNE_OPTS + 7)
#define OPT_TRAJ (NUM_COSYNE_OPTS + 8)
#define OPT_GAIN (NUM_COSYNE_OPTS + 9)
#define OPT_INC (NUM_COSYNE_OPTS + 10)

static struct cartArgs cartargs;

static struct argp_option env_options[] = {
    {"nm", OPT_NM, 0, 0, "Non-Markov"},
    {"poles", OPT_NPOLES, "P", 0, "Number of poles"},
    {"pl", OPT_PL, "L", 0, "Length of short pole"},
    {"pa", OPT_ANGLE, "A", 0, "initial angle of long pole in degrees"},
    {"noise", OPT_NOISE, "N", 0, "Sensor noise"},
    {"gruau", OPT_GRUAU, 0, 0, "Gruau fitness"},
    {"model", OPT_MODEL, "Model file", 0, "Use NN model for Environment."},
    {"traj", OPT_TRAJ, "T", 0, "Trajectory noise"},
    {"gain", OPT_GAIN, "G", 0, "Transfer correction gain"},
    {"inc", OPT_INC, 0, 0, "Evolve incrementally"},
    {0}};

static error_t
parse_env_opt(int key, char *arg, struct argp_state *state)
{
  //struct cosyneargs *ea = (struct cosyneargs *) state.input;
  //struct cartArgs *arguments = (struct cartArgs *) ea.envargs;

  switch (key)
  {
  case OPT_NM:
    cartargs.markov = false;
    break;
  case OPT_NPOLES:
    cartargs.numpoles = atoi(arg);
    break;
  case OPT_PL:
    cartargs.polelength = atof(arg);
    break;
  case OPT_NOISE:
    cartargs.noise = atof(arg);
    break;
  case OPT_ANGLE:
    cartargs.poleangle = atof(arg);
    break;
  case OPT_GRUAU:
    cartargs.gruau = true;
    break;
  case OPT_MODEL:
    cartargs.model = true;
    strcpy(cartargs.modelfile, arg);
    break;
  case OPT_TRAJ:
    cartargs.trajnoise = atof(arg);
    break;
  case OPT_GAIN:
    cartargs.gain = atof(arg);
    break;
  case OPT_INC:
    cartargs.incremental = true;
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

char header[] = "Cart Pole options:";

#endif

/////////////////////////////////////////////////////////////////

#ifdef _MAZE89
#define OPT_TRIALS (NUM_COSYNE_OPTS + 2)
#define OPT_INC (NUM_COSYNE_OPTS + 3)
#define OPT_GREEDY (NUM_COSYNE_OPTS + 4)
#define OPT_DEBUG (NUM_COSYNE_OPTS + 5)
#define OPT_NOISE (NUM_COSYNE_OPTS + 6)

static struct MazeArgs mazeargs;

static struct argp_option env_options[] = {
    {"trials", OPT_TRIALS, "T", 0, "number of trials"},
    {"inc", OPT_INC, 0, 0, "Evolve incrementally"},
    {"greedy", OPT_GREEDY, 0, 0, "greedy policy"},
    {"noise", OPT_NOISE, "N", 0, "Sensor noise"},
    {"debug", OPT_DEBUG, 0, 0, "dump"},
    {0}};

static error_t
parse_env_opt(int key, char *arg, struct argp_state *state)
{
  //struct espArgs *ea = (struct espArgs *) state.input;
  //struct cartArgs *arguments = (struct cartArgs *) ea.envargs;

  switch (key)
  {
  case OPT_TRIALS:
    mazeargs.trials = atoi(arg);
    break;
  case OPT_INC:
    mazeargs.incremental = true;
    break;
  case OPT_GREEDY:
    mazeargs.greedy = true;
    break;
  case OPT_NOISE:
    mazeargs.noise = atof(arg);
    break;
  case OPT_DEBUG:
    mazeargs.debug = true;
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}
char header[] = "Maze89 options:";

#endif

/////////////////////////////////////////////////////////////////

#ifdef _BLOCK
#define OPT_INC (NUM_COSYNE_OPTS + 1)
#define OPT_TASK (NUM_COSYNE_OPTS + 2)
#define OPT_TRIALS (NUM_COSYNE_OPTS + 3)
#define OPT_DISPLAY (NUM_COSYNE_OPTS + 4)
#define OPT_THINK (NUM_COSYNE_OPTS + 5)
#define OPT_SIZE (NUM_COSYNE_OPTS + 6)
#define OPT_STEPS (NUM_COSYNE_OPTS + 7)
#define OPT_BLOCKS (NUM_COSYNE_OPTS + 8)

static struct blockArgs blockargs;

static struct argp_option env_options[] = {
    //  {"lag"  , OPT_LAG,   "M"   , 0, "time lag"},
    {"inc", OPT_INC, 0, 0, "Evolve incrementally"},
    {"task", OPT_TASK, "int", 0, "Start with this many blocks"},
    {"trials", OPT_TRIALS, "t", 0, "Number of trials"},
    {"display", OPT_DISPLAY, 0, 0, "Display behavior"},
    {"think", OPT_THINK, 0, 0, "Allow 'think' moves"},
    {"size", OPT_SIZE, "int [6-10]", 0, "world size"},
    {"steps", OPT_STEPS, "int", 0, "Steps/Trial"},
    {"blocks", OPT_BLOCKS, "int", 0, "number of blocks"},
    {0}};

static error_t
parse_env_opt(int key, char *arg, struct argp_state *state)
{
  //struct espArgs *ea = (struct espArgs *) state.input;
  //struct cartArgs *arguments = (struct cartArgs *) ea.envargs;

  switch (key)
  {
  case OPT_INC:
    blockargs.incremental = true;
    break;
  case OPT_TRIALS:
    blockargs.trials = atoi(arg);
    break;
  case OPT_TASK:
    blockargs.task = atoi(arg);
    break;
  case OPT_DISPLAY:
    blockargs.display = true;
    break;
  case OPT_THINK:
    blockargs.think = true;
    break;
  case OPT_SIZE:
    blockargs.size = atoi(arg);
    break;
  case OPT_STEPS:
    blockargs.steps = atoi(arg);
    break;
  case OPT_BLOCKS:
    blockargs.blocks = atoi(arg);
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}
char header[] = "BlockPacker:";

#endif

/////////////////////////////////////////////////////////////////

static struct argp env_argp = {env_options, parse_env_opt, 0, 0};
static struct argp_child env_parser = {&env_argp, 0, header, 0};
static struct argp_child argp_children[2] = {env_parser};
static struct argp esp_parser = {options, parse_opt, args_doc, doc,
                                 argp_children};

//----------------------------------------------------------------------
// reseed random fns.
void reseed(int val)
{
  unsigned short seed[3];

  seed[0] = val;
  seed[1] = val + 1;
  seed[2] = val + 2;
  seed48(seed);
  srand48(val);
}

//////////////////////////////////////////////////////////////////////
//
// main
//
//////////////////////////////////////////////////////////////////////

#define RESTART false
int main(int argc, char *argv[])
{
  // Parse command-line arguments.

  //defaults
  maxGenerations = 5000;
  maxEvaluations = 10000000;

  cosyneargs.numpops = 5;
  cosyneargs.popsize = 40;
  cosyneargs.numevals = 1;
  cosyneargs.nettype = 3; //default to FullyRecurrentNetwork
  cosyneargs.net = genNet(intToNetEnum(cosyneargs.nettype), 0, 0, 0);
  cosyneargs.netlevel = false;
  cosyneargs.netlevelsize = 20;
  cosyneargs.grownets = true; //change to true once fixed
  cosyneargs.seed = time(NULL);
  cosyneargs.mutation = 0.8;
  cosyneargs.stagnation = 20;
  TEST = false;
  LESION = false;
  cosyneargs.verbose = false;
  cosyneargs.seednet = false;
  cosyneargs.paramfile = "";
  cosyneargs.savenets = false;
  cosyneargs.perc = false;
  cosyneargs.init_popsize = 0;
  cosyneargs.loci = 1;
  cosyneargs.neutral = 0.0;
  cosyneargs.wtrange = 10.0;
  cosyneargs.crossover_type = 3;

  //Weight prob params
  cosyneargs.proportional = false;
  cosyneargs.shift = 0.1;

#ifdef _CARTPOLE
  cartargs.markov = true;
  cartargs.numpoles = 2;
  cartargs.poleangle = 4.0156035; //degrees
  cartargs.polelength = 0.1;
  cartargs.noise = 0.0;
  cartargs.trajnoise = 0.0;
  cartargs.model = false;
  cartargs.incremental = false;
  cartargs.gruau = false;
#endif
#ifdef _BLOCK
  blockargs.incremental = false;
  blockargs.task = 0;
  blockargs.blocks = 6;
  blockargs.trials = 100;
  blockargs.think = false;
  //  blockargs.think_limit     = 8;
  blockargs.size = 6;
  blockargs.steps = 80;
#endif

#ifdef _ARGP
  argp_parse(&esp_parser, argc, argv, 0, 0, 0);
#endif

  reseed(cosyneargs.seed);
  cout << "Random seed : " << cosyneargs.seed << endl;

  if (!cosyneargs.init_popsize)
    cosyneargs.init_popsize = cosyneargs.popsize;

#ifdef _MAZE89
  Maze89 env(&mazeargs);
#endif
#ifdef _CARTPOLE
  CartPole env(&cartargs);
#endif
#ifdef _ROCKET
  Rocket env(&rocketargs);
#endif
#ifdef _BLOCK
  BlockPacker env(&blockargs);
#endif

  setUpCtrlC();

  //Create CoSyNE object
  CoSyNE cosyne(&cosyneargs, env, *cosyneargs.net);

  //if there is a parameter/population file
  if (cosyneargs.paramfile != "")
  { //!< load it. start evolving from it
    cosyne.loadFile(cosyneargs.paramfile.c_str());
  }
  //if we want to seed the population with a saved network
  else if (cosyneargs.seednet)
  {
    Network *net = loadNetworkXMLFile(NETFILE);
    cosyne.generatePopulationFromNetwork(net);
  }
  //test a saved network
  else if (TEST)
  {
    testNetwork(&env, NETFILE, LESION, true);
    exit(1);
  }
  //if not, initialize CoSyNE with command line parameters
  else
  {
    cosyne.init();
  }

  //  esp.savePopulation("out");
  if (RESTART)
  {
    cout << "RESTART BROKEN" << endl; //      esp.restart();
    exit(1);
  }
  else
    cosyne.evolve(maxGenerations, maxEvaluations);
  cosyne.endEvolution();
  exit(2);
}
