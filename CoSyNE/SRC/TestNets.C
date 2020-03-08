/////////////////////////////////////////////////////////////////////
//
//  Functions for testing Networks 
//  Copyright (C) 2007  Faustino J. Gomez
//
/////////////////////////////////////////////////////////////////////

#include <deque>
using std::deque;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <fstream>
using std::ifstream;
using std::ofstream;


#include "Environment.h"
#include "Network.h"
#include "Neuron.h"

#ifdef _XML
#ifdef _HCOSYNE
#include "HCoSyNE_SAX_Handlers.h"
#else
#include "CoSyNE_SAX_Handlers.h"
#endif
#endif

#include "TestNets.h"

#ifdef _XML
Network* loadNetworkXMLFile(const char* fname)
{ 
  ifstream fin(fname);

  if(fin.fail()){
    cerr << "ERROR: Could not open file: " << fname << endl;
    exit(1);
  }
  // Initialize the XML4C2 system
  try {
    XMLPlatformUtils::Initialize();
  }
  catch (const XMLException& toCatch){
    XERCES_STD_QUALIFIER cerr << "ERROR" << XERCES_STD_QUALIFIER endl;
  }

  SAXParser* parser = new SAXParser;
#ifdef _HCOSYNE
  HCoSyNE_SAXHandlers* handler = new HCoSyNE_SAXHandlers();
#else
  CoSyNE_SAXHandlers* handler = new CoSyNE_SAXHandlers();
#endif
  
  parser->setValidationScheme(SAXParser::Val_Always);
  parser->setValidationSchemaFullChecking(true);
  parser->setExitOnFirstFatalError(true);
  parser->setValidationConstraintFatal(true);

  parser->setDocumentHandler(handler);
  parser->setErrorHandler(handler);
  //parser->setDoValidation(true);
 
  cout << "Loading network file: " << fname << endl;
  parser->parse(fname);

  return handler->getNet();
}
#else

Network* loadNetworkXMLFile(const char* fname)
{
  cout << "TestNets.C: XML NOT ACTIVE" << endl;
}
#endif

Network* loadNetwork(Environment *e, char *filename)
{
  int i, j, type, numInputs, numUnits, numOutputs;
  Network *net;
  double tmp;
  char space;
  ifstream netfile(filename);

  netfile.precision(15);

  if (!netfile) {
    cerr << "Error - cannot open" << filename << " (in loadNetwork)" << endl;
    exit(1);
  }
  else{
    netfile >> type;
    netfile >> numInputs;
    netfile >> numUnits;
    netfile >> numOutputs;

    //    net = genNet(type, numInputs, numUnits, numOutputs);

    cout << "loading "  << filename << endl; 
    // cout << "type " << net->getName() << endl;
    cout << "topology " << net->getNumInputs() << " "
	 << net->getNumNeurons() << " "
	 << net->getNumOutputs() << endl;
    
    if( e->getInputDimension() != numInputs ||
	e->getOutputDimension() != numOutputs ){
      cerr << "Network " << filename 
	   << " has incorrect I/O dimensionality for " 
	   << e->getName() << " domain" << endl; 
      exit(1);
    }
    net->create();

    //NOTE networks are store in NUM_LOCI = 1 format           
    //make the following a call to Network::load();
    for(i=0;i<net->getNumNeurons();++i) {
      for(j=0;j<net->getGeneSize();++j) {
	netfile >> tmp;
	//fscanf(fptr, "%.25f%c", &tmp, &space);
	net->getNeuron(i)->setWeight(j, tmp);
      }
    }
  }
  
  netfile.close();
  //  cout << *net << endl;
  return net;
}


Network* loadNetwork(char *filename)
{
  int i, j, type, numInputs, numUnits, numOutputs;
  Network *net;
  double tmp;
  ifstream netfile(filename);


  if (!netfile) {
    cerr << "Error - cannot open" << filename << " (in loadNetwork)" << endl;
    exit(1);
  }
  else{
    netfile >> type;
    netfile >> numInputs;
    netfile >> numUnits;
    netfile >> numOutputs;

    //    net = genNet(type, numInputs, numUnits, numOutputs);

    cout << "loading "  << filename << endl; 
    //    cout << "type " << net->getName() << endl;
    cout << "topology " << net->getNumInputs() << " "
	 << net->getNumNeurons() << " "
	 << net->getNumOutputs() << endl;
    net->create();
          
    //make the following a call to Network::load();
    for(i=0;i<net->getNumNeurons();++i) {
      for(j=0;j<net->getGeneSize();++j) {
	netfile >> tmp;
	//fscanf(fptr, "%.25f%c", &tmp, &space);
	net->getNeuron(i)->setWeight(j, tmp);
      }
    }
  }
  
  netfile.close();
  //  cout << *net << endl;
  return net;
}
  


void testNetwork(Environment *e, char *filename, bool lesion, bool lesionWts)
{  
  Network *net;
  //  ofstream dotfile(prunedFile.c_str());
#ifdef _XML
  net = loadNetworkXMLFile(filename);
#else
  net = loadNetwork(filename);
#endif
  cout << "Testing " << filename << endl;
  testNetwork(e, net, lesion, lesionWts);
}


void testNetwork(Environment *e, Network *net, bool lesion, bool lesionWts)
{
  double tmp, unlesioned, afterPruningFit;
  double total;
  deque<int> lesions(0);
  std::string prunedFile("net");
  prunedFile += net->getID();
  ofstream outfile;

  Network *tmpNet = net->clone();
  *tmpNet = *net;

  //cout << *net << endl;
  //  tmp = e->generalizationTest(net);    

  for(int obj = 0; obj < e->getNumObjectives(); ++obj){
    e->setObjective(obj);
    //cout << *net;
    unlesioned = e->evaluateNetwork(net);
    //e->generalizationTest(net);    

    cout << " unlesioned fitness for objective " << obj << " = " << unlesioned << endl;
    
    if(lesion){
      for(int i = 0; i < net->getNumNeurons(); ++i){
	net->lesion(i);
	tmp = e->evaluateNetwork(net);
	//	e->generalizationTest(net);    

	cout << "Testing with neuron " << i << " lesioned: " << tmp << endl;
	
	if(tmp >= unlesioned * 0.99) //e->maxFitness)
	  lesions.push_front(i);
	//      tmp = e->generalizationTest(net);    
	net->unlesion();
      }
      if(lesions.size()){
	for(int i = 0; i < (int) lesions.size(); ++i){
	  //cout << lesions[i] << endl;
	  net->removeNeuron(lesions[i]);
	}
	prunedFile += ".pruned";
	//outfile.open(prunedFile);
	afterPruningFit = e->evaluateNetwork(net);

	cout << "Testing after pruning; fitness = " 
	     << afterPruningFit << endl;

	//	outfile << *net;
	if(afterPruningFit >= unlesioned) 
	  {}//NOTE testNetwork(e, net, lesion);//recurse
	else 
	  *net = *tmpNet;
      }
    }
  }
  //  delete net; 
  
  /*
  if(lesionWts){
    int j, wtCount = lesions.size() * net->getGeneSize();
    double wt;
    
    //    net = loadNetwork(e, (char*) prunedFile.c_str());
    for(int i = 0; i < net->getNumNeurons(); ++i){
      for(j = 0; j < net->getGeneSize(); ++j){
	if(net->getNeuron(i)->getWeight(j) != 0){
	  wt = net->getNeuron(i)->getWeight(j);
	  
	  net->getNeuron(i)->setWeight(j, 0.0);
	  
	  
	  tmp = e->evaluateNetwork(net);
	  
	  //	  net->getNeuron(i)->setWeight(j, wt);
	  if(tmp->getFitness(obj) >= unlesioned->getFitness(obj) * 0.99){ //e->maxFitness){
	    unlesioned = tmp;
	    ++wtCount;
	    cout << wtCount << endl;
	    
	    //e->generalizationTest(net);    
	  }
	  else
	    net->getNeuron(i)->setWeight(j, wt);
	  
	  cout << "Testing with neuron " << i+1 
	       << " weight " << j+1 << " (" << wt << ") " 
	       << " lesioned: " << tmp << endl;
	  
	}
      }
    }
  
  net->resetFitness();
    
  cout << (double) wtCount / ((net->getNumNeurons()+lesions.size()) * net->getGeneSize()) * 100
	 << " percent of weights pruned" 
	 << " fitness " 
	 << e->evaluateNetwork(net) << endl;
    
    //prunedFile += ".prunedWT";
    //cout << *net << endl;
    //net->saveText((char*) prunedFile.c_str());

  }
  */
}



