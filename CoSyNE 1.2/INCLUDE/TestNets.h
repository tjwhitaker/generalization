#ifndef _TEST_H_
#define _TEST_H_

class Environment;
class Network;



void testNetwork(Environment *, char *, bool lesion = false, bool lesionWts = false);
void testNetwork(Environment *, Network *, bool lesion = false, bool lesionWts = false);
Network* loadNetworkXMLFile(const char *);
Network* loadNetwork(Environment *e, char *);
Network* loadNetwork(char *);

#endif
