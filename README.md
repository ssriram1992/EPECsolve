# EPECcode
Code to compute mixed-equilibrium in linear EPECs.

*Manuscript in preparation. Link will be shared once ready*

# Prerequisites
- [Armadillo](http://arma.sourceforge.net/) (Version 9.2 or later)
- [Gurobi](https://www.gurobi.com/registration/download-reg) (Version 8.1 or later)
- [GNU make](https://www.gnu.org/software/make/)
- [gcc/g++](https://gcc.gnu.org/) (Tested on version 7.3. Must support C++11)

# Running
- Open `Makefile`. 
- Enter the path of your armadillo-installation in the line defining the variable `ARMA`. Typically, the location would be like `/opt/armadillo-code`.
- Enter the path of your Gurobi-installation in the line defining the variable `GUR`. Typically, the location would be like `/opt/gurobi/gurobi801/<Your OS>`.
- Run `make compileEPEC` to compile. 
- Run `make` to run the code.

