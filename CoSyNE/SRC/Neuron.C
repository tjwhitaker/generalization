//////////////////////////////////////////////////////////////////////
//
// Neuron for ESPWL
//
//////////////////////////////////////////////////////////////////////

#include <math.h>
#include "Neuron.h"


int NUM_LOCI = 1;
//extern int Fitness::numObjectives;

std::ostream& operator<<(std::ostream& os, Neuron &n)
{
  os.precision(15);
  os << "<" << n.getName() << ">" << endl;
  os << "    <WEIGHTS>" << endl;
  //os << n.getName() << " " << n.getID() << ": " << endl;
  for(unsigned int i=0; i <  n.weight.size(); ++i)
    os << n.weight[i]->w << " ";
  //NOTE
  os << endl;
  os << "    </WEIGHTS>" << endl;
  os << "    <FITNESS>";
  os << n.getFitness();
  os << "</FITNESS>" << endl;
  os << "</" << n.getName() << ">" << endl;
  return os;
}
/*
std::ostream& operator<<(std::ostream& os, Neuron &n)
{
  os.precision(5);
  os << n.getName() << " " << n.getID() << ": " << endl;
  for(unsigned int i=0; i <  n.weight.size(); ++i)
    os << n.weight[i]->w << " " << n.weight[i]->fitness << "  ";
  //NOTE
  os << endl;
  //  for(unsigned int i=0; i < n.fitnessVec.size(); ++i)
  //    os << n.fitnessVec[i] << "  ";
  
  os << endl;
  return os;
}

*/


//--------------------------------------------------------------------
//Neuron constructor
Neuron::Neuron(int size, double wtrange)
  : Neural(),
    lesioned(false),
    weight(size*NUM_LOCI),
    parent1(-1),
    parent2(-1),
    //fitness(new Fitness()),
    tag(false),
    frozen(false),
    elite(false),
    rank(0),
    p(0.0),
    wtRange(wtrange),
    created(false)
    //  numLoci(NUM_LOCI)
{

  name = "basic_neuron";
  static int counter = 0;
  id = ++counter;  //newID();
}


Neuron::~Neuron()
{
  for(unsigned int i = 0; i < getSize(); ++i){
    //    cout << "DELETING weight " << i << endl;
    delete weight[i];
  }
}

void Neuron::releaseWeights()
{  
  for(unsigned int i = 0; i < getSize(); ++i) 
    weight[i] = NULL;
  delete this;
}


//inline 
void Neuron::copyWeight(int i, Neuron *n) 
{ 
  if( checkBounds(i) &&  !frozen){ 
    weight[i]->w = n->weight[i]->w;
    weight[i+1]->w = n->weight[i+i]->w;
    resetFitness();  //size weight has been changed, fitness is no longer valid.
    newID(); 
  }
}



//inline
void Neuron::setWeight(int i, double w)
{ 
  if( checkBounds(i) &&  !frozen){ 
    //    double tmp = drand48();
    for (int j = 0; j < NUM_LOCI; ++j)
      //      weight[i * NUM_LOCI + j]->w = (double) w /NUM_LOCI;
      weight[i * NUM_LOCI + j]->w = w;
    resetFitness();  //size weight has been changed, fitness is no longer valid.
    newID(); 
  }
}

//inline 
void Neuron::setWeight(int i, Weight *w) 
{ 
  if( checkBounds(i) &&  !frozen){ 
    double tmp = drand48();
    for (int j = 0; j < NUM_LOCI; ++j)
      //      weight[i * NUM_LOCI + j]->w = (double) w /NUM_LOCI;
      weight[i * NUM_LOCI + j] = w;

    //weight[i] = w * tmp;
    //weight[i+1] = w - weight[i];
    
    //if(w != (weight[i] + weight[i+1]))
    //printf("ASDFS %.50f %.50f %.50f \n", w , weight[i] + weight[i+1], w - (weight[i] + weight[i+1]) );
      
    
    
    resetFitness();  //size weight has been changed, fitness is no longer valid.
    newID(); 
  }
}

//inline
void Neuron::setAllele(int i, double w) 
{ 
  if( i < weight.size()  &&  !frozen){ 
    weight[i]->w = w;
    resetFitness();  //weight has been changed, fitness is no longer valid.
    newID(); 
  }
}


//! Assign fitness to a Neuron.
void Neuron::assignFitness(double fit, int obj)
{
  fitness->setFitness(fit, obj);
  for(unsigned int i = 0 ; i < getSize(); ++i ){
    weight[i]->setFitness(fit, obj);
  }
}

//---------------------------------------------------------------------
//! Perturb the weights of a Neuron. 
/*! Used to search in a neighborhood around some Neuron (best).
 */
void Neuron::perturb(Neuron *n, double (*randFn)(double), double coeff)
{
  //WAS and WORKED
  //  for(unsigned int i = 0; i < weight.size(); ++i) 
  // weight[i] =  n->weight[i] + (randFn)(coeff);
  for(unsigned int i = 0; i < getSize(); ++i) 
    setWeight(i, n->getWeight(i) + (randFn)(coeff));


  //try using set weight later i.e.
  /*  int j;
  double w, tmp;
  for(unsigned int i = 0; i < weight.size() / NUM_LOCI; ++i) {
    tmp = drand48();
    w = n->getWeight(i);
    w += (randFn)(coeff);
    weight[i] = w * tmp;
    for(j = 1 ; j < NUM_LOCI ; ++j){
      w -= weight[i+j-1];
      weight[i+j] = w;
    }
  }
  */
  newID();
  resetFitness(); // reset the fitness after the weight vector is perturbed
}

void Neuron::perturb(Neuron *n, double coeff)
{
  perturb(n, rndCauchy, coeff);
}


//---------------------------------------------------------------------
//! Same as above but called on self and returns new Neuron
Neuron* Neuron::perturb(double coeff)
{
  Neuron *n = new Neuron( weight.size() );
  n->create();
  for(unsigned int i = 0 ; i < weight.size() ; ++i) 
    n->weight[i]->w = weight[i]->w + rndCauchy(coeff);
  return n;
}


//---------------------------------------------------------------------
//! Neuron assignment operator
Neuron& Neuron::operator=(const Neuron &n)
{  
  if(!n.created){
    cerr << "assigning uncreated Neuron; Neuron::operator=" << endl;
    abort();
  }
  //!< \todo NOTE
  id  = n.id;
  ///////////
 
  parent1 = n.parent1;
  parent2 = n.parent2;
  *fitness = *n.fitness;
  //weight = n.weight;
  
  if(!created) create();

  for(unsigned int i = 0; i < weight.size(); ++i){
    weight[i]->w = n.weight[i]->w;
    weight[i]->p = n.weight[i]->p;
  }
  lesioned = n.lesioned; //!< \todo NOTE

  return *this;
}

//! Check if two Neurons are equal
/*!
    Two Neurons are considered equal if they
    have equal \c weight vectors
*/
bool Neuron::operator==(Neuron &n)
{
  bool result = true;
  for(int i = 0; i < getSize(); ++i)
    if(weight[i]->w != n.weight[i]->w)
      result = false;

  return result;
}

//! Check if two Neurons are NOT equal.
/*!
    Two Neurons are considered equal if they
    have \c weight vectors that are NOT equal.
*/
bool Neuron::operator!=(Neuron &n)
{
  if(*this == n)
    return false;
  else
    return true;
}

//! Add a connection to a Neuron.
/*!
 */
inline
void Neuron::addConnection(int n)
{
  ///weight.insert(weight.begin() + n, 1.0);
}

inline 
void Neuron::removeConnection(int n)
{
  //weight.erase(weight.begin() + n);
}

//---------------------------------------------------------------------
//! Create a new set of random weights
void Neuron::create()
{
  double length = 0;
  double tmp;
  bool binary = false;
  static bool alert = binary;
  
  if(alert)
    cout << " BINARY BINARY BINARY!!!!!!!!!!!!!!" << endl;

  int j;
  for (unsigned int i = 0 ; i < getSize() ; ++i) {
    if(binary || alert){
      tmp = lrand48() % 2;
      alert = false;
    }
    else
      tmp = rndCauchy(wtRange);
    //tmp = (drand48() - 0.5) * wtRange * 2.0;

    for(j = 0 ; j < NUM_LOCI ; ++j) {
      weight[i*NUM_LOCI+j] = new Weight();
      weight[i*NUM_LOCI+j]->w = tmp/NUM_LOCI;
    }
    //NOTE this worked for the rocket: [i+j]
    //    weight[i+j] = tmp/NUM_LOCI;
    //length += weight[i] * weight[i];
  }
  //if(RELAX) weight [(int) weight.weight.size()()-1] = (double) (lrand48() % 3);
  //length = sqrt(length);
  //NOTE: normalization;
  // for (unsigned int i = 0 ; i < weight.size() ; ++i) 
  //weight[i] /= length;
  created = true;
}

void Neuron::mutate()
{
  weight[lrand48()%weight.size()]->w += rndCauchy( 0.3 );
}

void Neuron::mutate(double prob)
{
  if(drand48() < prob)
    weight[lrand48()%weight.size()]->w += rndCauchy( 0.3 );
}

//ONLY WORKS FOR FIRST SPLIT
void Neuron::split()
{
}
/*
  double tmp;
  int size = getSize(); 
  //cout << size << endl;
  //cout << weight.size() << endl;
  //  for (unsigned int i = 0 ; i < size ; ++i) 
  // cout << weight[i] << " " ;
  //cout << endl;

  for (unsigned int i = 0 ; i < size ; ++i) {
    
    tmp =  weight[i*2] * drand48();
    weight[i*2] -= tmp;
    weight.insert(weight.begin()+(i*2)+1, tmp);
  }
  //cout << weight.size() << endl << endl;
  
  //int k = NUM_LOCI;
  //NUM_LOCI *= 2;  
  //for (unsigned int i = 0 ; i < weight.size()/2 ; ++i) 
  //  cout << getWeight(i) << " " ;
  //cout << endl;
  //NUM_LOCI = k;
}
*/

//! perform neutral mutation by subtracting a random amount from
//! one of the weight's loci and adding it to the others 
void Neuron::mutateNeutral(double mutrate)
{
  if(NUM_LOCI > 1){
    if(drand48() < mutrate){
      //HAD THIS
      //int loci = lrand48() % weight()/NUM_LOCI;
      int loci = lrand48() % getSize();
      int tmp = lrand48() % NUM_LOCI;
      // double mut = drand48()*2.0 - 1.0;
      double mut = rndCauchy( 0.3 );
      //HAD THIS: WRONG but worked (?)
      //weight[loci+tmp] += mut;
      weight[loci*NUM_LOCI+tmp]->w += mut;
      for(int i = 0 ; i < NUM_LOCI ; ++i)
	if(i != tmp)
	  weight[loci*NUM_LOCI+i]->w -= mut;
	  //HAD THIS: WRONG but worked (?)
	  //weight[loci+i] -= mut;
    }
  }
}

double Neuron::getHashKey()
{
  double hk = 0;
  for (unsigned int i = 0 ; i < weight.size() ; ++i) 
    hk += weight[i]->w;

  return hk;
}
