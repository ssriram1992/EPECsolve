# Make file for Gurobi projects

# File name and output name
PROJECT=EPEC
FILEEPEC=BalasPolyhedron.o LCPtoLP.o EPEC.o 
OUTPUT=$(PROJECT)
ARGS=


# Armadillo stuff
ARMA=/opt/armadillo-code
ARMAINC=-I $(ARMA)/include
ARMALIB=-lblas -llapack
ARMAOPT=$(ARMAINC) $(ARMALIB)

# Gurobi stuff
GUR=/opt/gurobi/gurobi801/linux64
GURINC=-I $(GUR)/include 
GURLIB=-L $(GUR)/lib -lgurobi_c++ -lgurobi80 -lm 
GUROPT=$(GURINC) $(GURLIB)

# Generic objects not requiring changes
GCC=g++
OPTS=-fopenmp $(GUROPT) $(ARMAOPT) -O2 -Wall

runEPEC: compileEPEC
	./$(OUTPUT) $(ARGS)

compileEPEC: BalasPolyhedron.o LCPtoLP.o EPEC.o func.h
	$(GCC) $(FILEEPEC) $(OPTS) -o $(OUTPUT) 

BalasPolyhedron.o: func.h BalasPolyhedron.cpp
	$(GCC) -c BalasPolyhedron.cpp $(OPTS) 

LCPtoLP.o: func.h LCPtoLP.cpp
	$(GCC) -c LCPtoLP.cpp $(OPTS) 

EPEC.o: func.h EPEC.cpp
	$(GCC) -c EPEC.cpp $(OPTS)

clean:
	rm -rf $(OUTPUT)
