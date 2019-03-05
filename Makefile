# Make file for Gurobi projects

# File name and output name
PROJECT=EPEC
FILEEPEC=LCPtoLP.o EPEC.o Games.o LCPTree.o
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
OTHEROPTS=-fopenmp -O2 -Wall -std=c++11
OPTS= $(GUROPT) $(ARMAOPT)  $(OTHEROPTS)

runEPEC: compileEPEC
	./$(OUTPUT) $(ARGS)

compileEPEC: LCPtoLP.o func.h Games.o LCPTree.o EPEC.o 
	$(GCC) $(FILEEPEC) $(OPTS) -o $(OUTPUT) 

BalasPolyhedron.o: func.h BalasPolyhedron.cpp
	@echo "Compiling BalasPolyhedron.cpp omited"
#$(GCC) -c BalasPolyhedron.cpp $(OPTS) 

LCPtoLP.o: func.h LCPtoLP.cpp
	$(GCC) -c LCPtoLP.cpp $(OPTS) 

LCPTree.o: func.h LCPtoLP.o LCPTree.cpp
	$(GCC) -c LCPTree.cpp $(OPTS)

EPEC.o: func.h EPEC.cpp
	$(GCC) -c EPEC.cpp $(OPTS)

Games.o: func.h Games.cpp
	$(GCC) -c Games.cpp $(OPTS)

clean:
	rm -rf $(OUTPUT)
	rm -rf *.o

game: func.h Games.cpp
	$(GCC) -c Games.cpp $(OPTS) 
	$(GCC) Games.o LCPtoLP.o BalasPolyhedron.o func.h -o Games $(OPTS)
	./Games

sand: sand.o
	$(GCC) sand.o $(OPTS) -o sand
	./sand

sand.o: sand.cpp
	$(GCC) -c sand.cpp $(OPTS)


open: 
	vim -p func.h Games.cpp EPEC.cpp LCPtoLP.cpp LCPTree.cpp
