# File name and output name
EPEC_HOME=/home/sriram/Dropbox/code/EPEC/code
SRC=$(EPEC_HOME)/src
OBJ=$(EPEC_HOME)/obj
BIN=$(EPEC_HOME)/bin

PROJECT=EPEC
FILEEPEC=$(OBJ)/LCPtoLP.o $(OBJ)/EPEC.o $(OBJ)/Games.o $(OBJ)/Models.o
OUTPUT=$(BIN)/$(PROJECT)
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
	$(OUTPUT) $(ARGS)

valgrind: compileEPEC
	valgrind --leak-check=full --show-leak-kinds=all  -v ./$(OUTPUT) $(ARGS)

compileEPEC: EPEC

EPEC: $(FILEEPEC)
	$(GCC) $(FILEEPEC) $(OPTS) -o $(OUTPUT) 

$(OBJ)/LCPtoLP.o: $(SRC)/epecsolve.h $(SRC)/lcptolp.h $(SRC)/LCPtoLP.cpp
	$(GCC) -c $(SRC)/LCPtoLP.cpp $(OPTS) -o $(OBJ)/LCPtoLP.o

$(OBJ)/EPEC.o: $(SRC)/epecsolve.h $(SRC)/models.h $(SRC)/EPEC.cpp
	$(GCC) -c $(SRC)/EPEC.cpp $(OPTS) -o $(OBJ)/EPEC.o

$(OBJ)/Games.o: $(SRC)/epecsolve.h $(SRC)/games.h $(SRC)/Games.cpp
	$(GCC) -c $(SRC)/Games.cpp $(OPTS) -o $(OBJ)/Games.o

$(OBJ)/Models.o: $(SRC)/epecsolve.h $(SRC)/models.h $(SRC)/Models.cpp
	$(GCC) -c $(SRC)/Models.cpp $(OPTS) -o $(OBJ)/Models.o

clean:
	rm -rf $(OUTPUT)
	rm -rf $(OBJ)/*.o

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
	doxygen docs/refConf

docDetailed:
	doxygen docs/refDetConf

edit: 
	vim -p epecsolve.h Games.cpp LCPtoLP.cpp  Models.cpp EPEC.cpp

tag:
	ctags *.cpp *.h
	@echo "All tags done. Use Ctrl+] to follow a tag in vim and Ctrl+O to go back"


