# Make file for Gurobi projects

# File name and output name
PROJECT=EPEC
FILEEPEC=LCPtoLP.o EPEC.o Games.o Models.o
OUTPUT=$(PROJECT)
ARGS=


# Armadillo stuff
ARMA=/opt/armadillo-code
ARMAINC=-I $(ARMA)/include
ARMALIB=-lblas -llapack
ARMAOPT=$(ARMAINC) $(ARMALIB)

# Gurobi stuff
# GUR=/opt/gurobi810/linux64
GUR=/opt/gurobi/gurobi801/linux64
GURINC=-I $(GUR)/include 
GURLIB=-L $(GUR)/lib -lgurobi_c++ -lgurobi80 -lm 
# GURLIB=-L $(GUR)/lib -lgurobi_c++ -lgurobi81 -lm 
GUROPT=$(GURINC) $(GURLIB)

# Generic objects not requiring changes
GCC=g++
OTHEROPTS= -O2 -Wall -std=c++11
OPTS= $(GUROPT) $(ARMAOPT)  $(OTHEROPTS)

runEPEC: compileEPEC
	./$(OUTPUT) $(ARGS)

valgrind: compileEPEC
	valgrind --leak-check=full --show-leak-kinds=all  -v ./$(OUTPUT) $(ARGS)

compileEPEC: LCPtoLP.o func.h Games.o Models.o EPEC.o 
	$(GCC) $(FILEEPEC) $(OPTS) -o $(OUTPUT) 

BalasPolyhedron.o: func.h BalasPolyhedron.cpp
	@echo "Compiling BalasPolyhedron.cpp omited"
#$(GCC) -c BalasPolyhedron.cpp $(OPTS) 

LCPtoLP.o: func.h LCPtoLP.cpp
	$(GCC) -c LCPtoLP.cpp $(OPTS) 

# LCPTree.o: func.h LCPTree.cpp
# $(GCC) -c LCPTree.cpp $(OPTS)

EPEC.o: func.h EPEC.cpp
	$(GCC) -c EPEC.cpp $(OPTS)

Games.o: func.h Games.cpp
	$(GCC) -c Games.cpp $(OPTS)

Models.o: func.h Models.cpp
	$(GCC) -c Models.cpp $(OPTS)

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

docSimple:
	doxygen refConf

docDetailed:
	doxygen refDetConf

edit: 
	vim -p func.h Games.cpp LCPtoLP.cpp  Models.cpp EPEC.cpp

tag:
	ctags *.cpp *.h
	@echo "All tags done. Use Ctrl+] to follow a tag in vim and Ctrl+O to go back"
