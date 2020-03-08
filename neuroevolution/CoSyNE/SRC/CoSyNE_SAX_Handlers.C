/*
 *  CoSyNE_SAX_Handler.C
 *
 *  Xerces C++ SAX Handlers for parsing CoSyNE populations and networks
 */

// ---------------------------------------------------------------------------
//  Includes
// ---------------------------------------------------------------------------
#include <algorithm>
//using std::reverse;
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include "Neuron.h"
#include "Network.h"
//NOTE was CoSyNE-NCD.h
#include "CoSyNE.h"
#include "CoSyNE_SAX_Handlers.h"
#include <xercesc/sax/AttributeList.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/util/XMLStringTokenizer.hpp>



// ---------------------------------------------------------------------------
//  CoSyNE_SAXHandlers: Constructors and Destructor
// ---------------------------------------------------------------------------
CoSyNE_SAXHandlers::CoSyNE_SAXHandlers(CoSyNE *_cosyne) 
  : cosyne(_cosyne),
    fSawErrors(false), 
    subPop(0),
    neuron(0),
    isNetwork(false)
{
}

CoSyNE_SAXHandlers::CoSyNE_SAXHandlers() 
  : cosyne(NULL), 
    fSawErrors(false), 
    subPop(0),
    neuron(0),
    isNetwork(false)
{
}


CoSyNE_SAXHandlers::~CoSyNE_SAXHandlers()
{
}


// ---------------------------------------------------------------------------
//  CoSyNE_SAXHandlers: Implementation of the SAX DocumentHandler interface
// ---------------------------------------------------------------------------
void CoSyNE_SAXHandlers::startElement(const   XMLCh* const    name,      
				   AttributeList&  attributes)
{
  //  cout << XMLString::transcode(name) << endl;
  element = XMLString::transcode(name);
  if(!strcmp("POPULATION", element)){
    //cosyne->subPopSize = atoi(XMLString::transcode(attributes.getValue("size") ) ) ;
    //cout << XMLString::transcode(attributes.getValue("size") ) << endl;
    cout << "loading subpopulation " << subPop+1 << "..." << endl;
  }
  if(!strcmp("NETWORK", element)){
    //create a network

    //cosyne->network
    net = genNet(XMLString::transcode(attributes.getValue("type")),
		 atoi( XMLString::transcode(attributes.getValue("inputs")) ),
		 atoi( XMLString::transcode(attributes.getValue("size")) ),
		 atoi( XMLString::transcode(attributes.getValue("outputs")) ) );

    
    cout << "Net type     : " << net->getName()  << endl;
    cout << "No. inputs   : " << net->getNumInputs() << endl;
    cout << "No. neurons  : " << net->getNumNeurons() << endl;
    cout << "No. outputs  : " << net->getNumOutputs()  << endl;
    isNetwork = true;
    
  }
}

void CoSyNE_SAXHandlers::endElement(const   XMLCh* const    name)   
{ 
  //cout << XMLString::transcode(name) << endl;
  if(!strcmp("PREAMBLE", XMLString::transcode(name)) ){
    cosyne->subPopSize = 0; //so that empty subpops are created
    cosyne->init();
    cout << "creating CoSyNE " << cosyne->numPops << endl;
  }
  else if(!strcmp("POPULATION", XMLString::transcode(name)) ){
    // reverse( cosyne->weightPops[subPop].begin(), cosyne->weightPops[subPop].end() );
    ++subPop;
    neuron = 0; 
  }
  else if(!strcmp("COSYNE", XMLString::transcode(name)) ){
    cosyne->subPopSize = cosyne->weightPops[0].size();
  }
  else if(!strcmp("NETWORK", XMLString::transcode(name)) ){
    net->created = true;
 }
  element = "";
}


void CoSyNE_SAXHandlers::characters( const XMLCh* const chars, 
				     const XMLSize_t 	length)
				     
{ 

  //cout << XMLString::transcode(chars) << endl;
  if(!strcmp("NET_TYPE", element)){
    cosyne->network = genNet(XMLString::transcode(chars),0,0,0);	
  }
  else if(!strcmp("GENERATION", element)){
    cosyne->generation = atoi(XMLString::transcode(chars));
  }
  else if(!strcmp("NUM_POPS", element)){
    cosyne->numPops = atoi(XMLString::transcode(chars));
  }
  else if(!strcmp("SUB_POP_SIZE", element)){
    cosyne->subPopSize = atoi(XMLString::transcode(chars));
  }
  else if(!strcmp("MUTATION", element)){
    cosyne->mutationRate = atof(XMLString::transcode(chars));
  }
  else if(!strcmp("NEVALS", element)){
    cosyne->numEvals = atoi(XMLString::transcode(chars));
  }
  else if(!strcmp("STAGNATION", element)){
    cosyne->stagnation = atoi(XMLString::transcode(chars));
  }
  else if(!strcmp("WT_RANGE", element)){
    cosyne->wtRange = atof(XMLString::transcode(chars));
  }  
  else if(!strcmp("ENVIRONMENT", element) && cosyne != NULL){
    if(strcmp(cosyne->Envt.getName().c_str(), XMLString::transcode(chars)) ){
       cerr << "ERROR: Wrong environment for this population file!" << endl;
       exit(1);
    }
  }
  //load the weight into a subpopulation
  else if(!strcmp("POPULATION", element)){
    XMLStringTokenizer *tokenizer = new XMLStringTokenizer(chars);
    XMLCh *token;
    while(tokenizer->hasMoreTokens()){
      token = tokenizer->nextToken();
      Weight *wt = new Weight();
      wt->w = atof(XMLString::transcode(token));
      cosyne->weightPops[subPop].push_back( wt );
    } 
    delete tokenizer;
  }
  //if it's a network load one neuron 
  else if(isNetwork && !strcmp("WEIGHTS", element)){
    Neuron *tmpNeuron = new Neuron(0);
    XMLStringTokenizer *tokenizer = new XMLStringTokenizer(chars);
    XMLCh *token;
    while(tokenizer->hasMoreTokens()){
      token = tokenizer->nextToken();
      Weight *wt = new Weight();
      wt->w = atof(XMLString::transcode(token));
      tmpNeuron->weight.push_back( wt );
    } 
    delete tokenizer;
    tmpNeuron->created = true; //NOTE make automatic
    net->setNeuron( tmpNeuron, neuron );
    neuron++;
  }
}

void CoSyNE_SAXHandlers::ignorableWhitespace( const   XMLCh* const chars, 
					      const unsigned int length)
{
}

// ---------------------------------------------------------------------------
//  CoSyNE_SAXHandlers: Overrides of the SAX ErrorHandler interface
// ---------------------------------------------------------------------------
void CoSyNE_SAXHandlers::error(const SAXParseException& e)
{
    fSawErrors = true;
    XERCES_STD_QUALIFIER cerr << "\nError at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void CoSyNE_SAXHandlers::fatalError(const SAXParseException& e)
{
    fSawErrors = true;
    XERCES_STD_QUALIFIER cerr << "\nFatal Error at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void CoSyNE_SAXHandlers::warning(const SAXParseException& e)
{
    XERCES_STD_QUALIFIER cerr << "\nWarning at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void CoSyNE_SAXHandlers::resetErrors()
{
    fSawErrors = false;
}
