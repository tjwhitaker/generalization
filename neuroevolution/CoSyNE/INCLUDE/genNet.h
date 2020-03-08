///////////////////////////////////////////////////////////////////
// genNet.h
// Copyright (C) 2007  Faustino J. Gomez
///////////////////////////////////////////////////////////////////

#ifndef _GEN_NET_H_
#define _GEN_NET_H_
enum EspNetType {       TYPE_FeedForwardNetwork,
      TYPE_SimpleRecurrentNetwork,
      TYPE_SecondOrderRecurrentNetwork,
      TYPE_FullyRecurrentNetwork,
      TYPE_FullyRecurrentNetwork2,
      TYPE_LinearRecurrentNetwork,
      TYPE_SingleLayerFeedForwardNetwork,
      TYPE_LSTMNetwork
};
class Network;
Network* genNet(char *type, int in, int hid, int out);
Network* genNet(EspNetType type, int in, int hid, int out);
EspNetType netStrNameToEnum(char *type);
char*  netEnumToStrName(EspNetType type);
EspNetType  intToNetEnum(int type);
char* getNetTypeName(Network *net);
void listNetTypes();

#endif
