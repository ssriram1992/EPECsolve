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
GUR=/opt/gurobi801/linux64
# GUR=/opt/gurobi/gurobi801/linux64
GURINC=-I $(GUR)/include 
GURLIB=-L $(GUR)/lib -lgurobi_c++ -lgurobi80 -lm 
# GURLIB=-L $(GUR)/lib -lgurobi_c++ -lgurobi81 -lm 
GUROPT=$(GURINC) $(GURLIB)

# Generic objects not requiring changes
# GCC=g++
GCC=g++-4.8
OTHEROPTS= -O2 -Wall -Wno-comment -std=c++11
OPTS= $(GUROPT) $(ARMAOPT)  $(OTHEROPTS)

runEPEC: compileEPEC
	./$(OUTPUT) $(ARGS)

valgrind: compileEPEC
	valgrind --leak-check=full --show-leak-kinds=all  -v ./$(OUTPUT) $(ARGS)

compileEPEC: EPEC

EPEC: LCPtoLP.o Games.o Models.o EPEC.o 
	$(GCC) $(FILEEPEC) $(OPTS) -o $(OUTPUT) 

LCPtoLP.o: epecsolve.h lcptolp.h LCPtoLP.cpp
	$(GCC) -c LCPtoLP.cpp $(OPTS) 

EPEC.o: epecsolve.h models.h EPEC.cpp
	$(GCC) -c EPEC.cpp $(OPTS)

Games.o: epecsolve.h games.h Games.cpp
	$(GCC) -c Games.cpp $(OPTS)

Models.o: epecsolve.h models.h Models.cpp
	$(GCC) -c Models.cpp $(OPTS)

clean:
	rm -rf $(OUTPUT)
	rm -rf *.o

sand: sand.o
	$(GCC) sand.o $(OPTS) -o sand
	./sand


sand2: sand2.o
	$(GCC) sand2.o $(OTHEROPTS) -o sand2
	./sand2

sand2.o: sand2.cpp
	$(GCC) -c sand2.cpp $(OPTS)


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


