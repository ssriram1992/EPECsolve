# File name and output name
# EPEC_HOME=/home/sanksrir/Documents/code/EPEC
EPEC_HOME=/home/sriram/Dropbox/code/EPEC/code
SRC=$(EPEC_HOME)/src
OBJ=$(EPEC_HOME)/obj
BIN=$(EPEC_HOME)/bin

PROJECT=EPEC
FILEEPEC=$(OBJ)/LCPtoLP.o $(OBJ)/Games.o $(OBJ)/Models.o $(OBJ)/Utils.o
OUTPUT=$(BIN)/$(PROJECT)
ARGS=

# Logging
# BOOST_HOME=/home/x86_64-unknown-linux_ol7-gnu/boost-1.70.0
BOOST_HOME=/home/sriram/Install/boost_1_70_0
# BOOST_LIB_D=$(BOOST_HOME)/lib/libboost_
BOOST_LIB_D=$(BOOST_HOME)/stage/lib/libboost_
# BOOSTLIB=$(BOOST_LIB_D)log.a $(BOOST_LIB_D)log_setup.a $(BOOST_LIB_D)unit_test_framework.a $(BOOST_LIB_D)system.a $(BOOST_LIB_D)thread.a $(BOOST_LIB_D)chrono.a  -lpthread $(BOOST_LIB_D)prg_exec_monitor.a
BOOSTLIB=$(BOOST_LIB_D)unit_test_framework.a $(BOOST_LIB_D)program_options.a  $(BOOST_LIB_D)log.a $(BOOST_LIB_D)log_setup.a $(BOOST_LIB_D)system.a $(BOOST_LIB_D)thread.a $(BOOST_LIB_D)chrono.a  -lpthread $(BOOST_LIB_D)prg_exec_monitor.a
# BOOSTOPT=-I $(BOOST_HOME)/include $(BOOSTLIB) 
BOOSTOPT=-I $(BOOST_HOME) 

# Armadillo stuff
ARMA=/opt/armadillo-code
# ARMAINC=-I $(ARMA)/include
ARMAINC=
# ARMALIB=-lblas -llapack
ARMALIB=-larmadillo
ARMAOPT=$(ARMAINC) 

# Gurobi stuff
GUR=/opt/gurobi811/linux64
# GUR=/home/gurobi/8.1.0/linux64
# GUR=/opt/gurobi/gurobi801/linux64
GURINC=-I $(GUR)/include 
# GURLIB=-L $(GUR)/lib -lgurobi_c++ -lgurobi80 -lm 
GURLIB= $(GUR)/lib/libgurobi_c++.a $(GUR)/lib/libgurobi81.so -lm  
# GURLIB=-L $(GUR)/lib -lgurobi_c++ -lgurobi81 -lm 
GUROPT=$(GURINC)

# Generic objects not requiring changes
# GCC=g++
GCC=g++-4.8
#OTHEROPTS= -O2 -std=c++11 -I include/
OTHEROPTS= -O3 -std=c++11 -I include/ #-Wall -Wextra -Wpedantic #-Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wconversion 
OPTS= $(GUROPT) $(ARMAOPT) $(OTHEROPTS) $(BOOSTOPT) 
LINKOPTS=$(GURLIB) $(ARMALIB) $(BOOSTLIB)

runEPEC: compileEPEC
	$(OUTPUT) $(ARGS)

valgrind: compileEPEC
	valgrind --leak-check=full --show-leak-kinds=all  -v ./$(OUTPUT) $(ARGS)

compileEPEC: $(OUTPUT)

makeInstances:  $(FILEEPEC) $(OBJ)/makeTests.o 
	@echo making the test cases...
	$(GCC) $(FILEEPEC) $(OBJ)/makeTests.o  $(OPTS) $(LINKOPTS) -o bin/genTestInstances
	./bin/genTestInstances

$(OUTPUT): $(FILEEPEC) $(OBJ)/EPEC.o 
	@echo Linking...
	$(GCC) $(FILEEPEC) $(OBJ)/EPEC.o  $(OPTS) $(LINKOPTS) -o $(OUTPUT) 

$(OBJ)/LCPtoLP.o: $(SRC)/epecsolve.h $(SRC)/lcptolp.h $(SRC)/LCPtoLP.cpp 
	$(GCC) -c $(SRC)/LCPtoLP.cpp $(OPTS) -o $(OBJ)/LCPtoLP.o

$(OBJ)/EPEC.o: $(SRC)/epecsolve.h $(SRC)/models.h $(SRC)/EPEC.cpp
	$(GCC) -c $(SRC)/EPEC.cpp $(OPTS) -o $(OBJ)/EPEC.o

$(OBJ)/Games.o: $(SRC)/epecsolve.h $(SRC)/games.h $(SRC)/Games.cpp
	$(GCC) -c $(SRC)/Games.cpp $(OPTS) -o $(OBJ)/Games.o

$(OBJ)/Models.o: $(SRC)/epecsolve.h $(SRC)/models.h $(SRC)/Models.cpp
	$(GCC) -c $(SRC)/Models.cpp $(OPTS) -o $(OBJ)/Models.o

$(OBJ)/Utils.o: $(SRC)/epecsolve.h $(SRC)/utils.h $(SRC)/Utils.cpp
	$(GCC) -c $(SRC)/Utils.cpp $(OPTS) -o $(OBJ)/Utils.o

$(OBJ)/makeTests.o: $(SRC)/epecsolve.h $(SRC)/models.h $(SRC)/makeTests.cpp
	$(GCC) -c $(SRC)/makeTests.cpp $(OPTS) -o $(OBJ)/makeTests.o

EPECtest: $(EPEC_HOME)/test/EPEC
	@echo "Starting the tests..."
	@$(EPEC_HOME)/test/EPEC -l success $(ARGS)
	@echo "Tests completed"

$(EPEC_HOME)/test/EPEC.o: $(EPEC_HOME)/test/EPEC.cpp
	@echo "Compiling the tests..."
	$(GCC) -c $(EPEC_HOME)/test/EPEC.cpp $(OPTS) $(BOOSTOPT) -o $(EPEC_HOME)/test/EPEC.o


$(EPEC_HOME)/test/EPEC: $(FILEEPEC) $(EPEC_HOME)/test/EPEC.o
	$(GCC) $(FILEEPEC) $(EPEC_HOME)/test/EPEC.o  $(BOOSTOPT) $(BOOSTLIB) $(OPTS) $(LINKOPTS) -o $(EPEC_HOME)/test/EPEC

clean:
	rm -rf $(OUTPUT) $(EPEC_HOME)/test/EPEC
	rm -rf $(OBJ)/*.o

instance:
	$(OUTPUT) -i dat/Instance -m 1 -w 2

format:
	@clang-format-8 -style=llvm -i src/*.cpp
	@clang-format-8 -style=llvm -i src/*.h

docSimple:
	doxygen docs/refConf

docDetailed:
	doxygen docs/refDetConf

edit: 
	vim -p src/epecsolve.h src/Games.cpp src/LCPtoLP.cpp  src/Models.cpp src/Utils.cpp src/EPEC.cpp

tag:
	ctags src/*.cpp src/*.h
	@echo "All tags done. Use Ctrl+] to follow a tag in vim and Ctrl+O to go back"

install:
	mkdir -p dat
	mkdir -p bin
	mkdir -p obj

$(BIN)/ChileArgentina: $(FILEEPEC) $(OBJ)/ChileArgentina.o 
	@echo Linking...
	$(GCC) $(FILEEPEC) $(OBJ)/ChileArgentina.o  $(OPTS) $(LINKOPTS) -o $(BIN)/ChileArgentina

$(OBJ)/ChileArgentina.o: $(SRC)/epecsolve.h $(SRC)/models.h $(SRC)/ChileArgentina.cpp
	$(GCC) -c $(SRC)/ChileArgentina.cpp $(OPTS) -o $(OBJ)/ChileArgentina.o

chile: $(BIN)/ChileArgentina
	$(BIN)/ChileArgentina

$(BIN)/example: $(FILEEPEC) $(OBJ)/example.o 
	@echo Linking...
	@$(GCC) $(FILEEPEC) $(OBJ)/example.o  $(OPTS) $(LINKOPTS) -o $(BIN)/example

$(OBJ)/example.o: $(SRC)/epecsolve.h $(SRC)/models.h $(SRC)/example.cpp
	@$(GCC) -c $(SRC)/example.cpp $(OPTS) -o $(OBJ)/example.o

example: $(BIN)/example
	@$(BIN)/example
