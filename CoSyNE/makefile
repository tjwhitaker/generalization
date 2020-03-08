######################################################################
#
#  CoSyNE 1.1 makefile 
#
######################################################################

CC = g++
LD = g++

DIR = .
INCLUDES = -I$(HOME)/include
CFLAGS =  $(OPT)  $(INCLUDES) -I$(DIR)/INCLUDE -I$(DIR)/SRC 
LDFLAGS = -L.
ARCHIVES = -lm -lstdc++


DEPENDFLAGS = $(INCLUDES)

TARGET = cosyne
LIBNAME = libesp.a

DEBUG = 
PROFILE = 
OPT=-O2

OBJECTS-NET = Network.o \
		FeedForwardNetwork.o \
		FullyRecurrentNetwork.o \
		SimpleRecurrentNetwork.o \
		SecondOrderRecurrentNetwork.o \
		LSTMNetwork.o \
		LinearRecurrentNetwork.o \
		SingleLayerFeedForwardNetwork.o

OBJECTS-AUX = signal-handler.o 
OBJECTS-ENV = CartPole.o 
OBJECTS     = NE.o Neuron.o Environment.o \
		$(OBJECTS-NET) $(OBJECTS-AUX)  ne-utils.o 


#shouldn't need anything like this, unless on a mac
ARGP = -I$(HOME)/argp-standalone-1.3 -D_ARGP
#ARCHIVES += -L/usr/include -largp

#######################################################################
# XML stuff
#######################################################################

#if no XML SAX parser comment out the next line
XML = -D_XML 

#you might need to change these paths to the xerces lib
ifdef XML   
XML_LIB = -L$(HOME)/Source/xerces-c_2_8_0-x86_64-linux-gcc_3_4/lib -lxerces-c
CFLAGS += -I$(HOME)/Source/xerces-c_2_8_0-x86_64-linux-gcc_3_4/include 

OBJECTS += CoSyNE_SAX_Handlers.o 

endif

ARCHIVES += $(XML_LIB) # -lesp

#########################################################################

# Implicit rules
OBJ/%.o : SRC/%.C INCLUDE/%.h  INCLUDE/Fitness.h 
	$(CC) -c $(CFLAGS) $(XML) $< -o $@ 


OBJ/genNet.o: $(OBJECTS-NET:%=OBJ/%) INCLUDE/Network.h create_net_types.pl
	./create_net_types.pl $(PWD);
	$(CC) -c $(CFLAGS) SRC/genNet.C -o $@


OBJ/TestNets.o: SRC/TestNets.C INCLUDE/TestNets.h 
	$(CC) -c $(CFLAGS) $(XML) SRC/TestNets.C -o $@

SRC/genNet.C: INCLUDE/Networks.h 
	./create_net_types.pl $(PWD);
	$(CC) -c $(CFLAGS) SRC/genNet.C -o OBJ/genNet.o


OBJ/main-cosyne-pole.o: SRC/main-cosyne.C
	$(CC) -c $(CFLAGS) $(ARGP) -D_CARTPOLE SRC/main-cosyne.C -o $@



#--------------------------------------------------------------------
# Cart Pole environment

cosyne-dpole: SRC/genNet.C makefile  $(OBJECTS:%=OBJ/%) \
	OBJ/CoSyNE.o OBJ/main-cosyne-pole.o OBJ/CartPole.o OBJ/TestNets.o
	$(LD) OBJ/TestNets.o OBJ/CoSyNE.o OBJ/genNet.o OBJ/main-cosyne-pole.o OBJ/CartPole.o \
	$(LDFLAGS) $(OBJECTS:%=OBJ/%) $(ARCHIVES) -o $@


#######################################################################

clean::	
	rm -f cosyne-dpole 
	rm -f OBJ/*.o
	rm -f SRC/genNet.C
	rm -f INCLUDE/genNet.h
	rm -f *~
	rm -f core


tar:	
	tar cvhf CoSyNE-1.2.tar --exclude='*.o' \
	--exclude 'genNet.*' \
	$(OBJECTS:%.o=SRC/%.C) \
	SRC/CoSyNE.C \
	INCLUDE/CoSyNE.h \
	$(OBJECTS:%.o=INCLUDE/%.h) \
	INCLUDE/Fitness.h \
	INCLUDE/Networks.h \
	INCLUDE/Neural.h \
	INCLUDE/Weight.h \
	INCLUDE/TestNets.h \
	SRC/TestNets.C \
	SRC/main-cosyne.C \
	INCLUDE/CartPole.h \
	SRC/CartPole.C \
	OBJ/ \
	create_net_types.pl \
	network.dtd \
	cosyne.dtd \
	makefile \
	README


# DO NOT DELETE THIS LINE -- make depend depends on it.
