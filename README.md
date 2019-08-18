# EPECcode
Code to compute mixed-equilibrium in linear EPECs.

*Manuscript in preparation. Link will be shared once ready*

# Prerequisites
- [Armadillo](http://arma.sourceforge.net/) (Version 9.6 or later)
	* BLAS
	* ARPACK
	* LAPACK
- [Gurobi](https://www.gurobi.com/registration/download-reg) (Version 8.1 or later)
- [gcc/g++](https://gcc.gnu.org/) (Tested on version 4.8. Must support C++11 and be compatible with your version of Gurobi) `sudo apt install gcc ` will install gcc/g++ on an Ubuntu machine.
- [GNU make](https://www.gnu.org/software/make/) `sudo apt install make` will install GNU make on an Ubuntu machine.
- [Rapid JSON](http://rapidjson.org/) To export results and save example problem instances
- [DOxygen](http://www.doxygen.nl) Only if you need documentation. `sudo apt install doxygen ` will install DOxygen on an Ubuntu machine.

# Getting the documentation
One can generate two versions of documentation for this project..
- Use the simple version of documentation, if you are only interested in using this as a predefined library which you don't intend to edit. This version of the documentation gives a sufficiently detailed explanation of every class or function you might every have to use. To avail this version, run
```bash
make docSimple
```
- Use the complete documentation if you are interested in every implementation detail of the code.This gives a complete description of every private member and fields in every class, all of which might be useful if you want to edit the code in here. To avail this version, run
```bash
make docDetailed
```
- You can alternatively use the [online documentation](https://ssriram1992.github.io/EPECsolve/html/index.html)

# Running
- Download the [project](https://github.com/ssriram1992/EPECsolve/). If you do not have access, please email [sriram.sankaranarayanan@polymtl.ca](mailto:sriram.sankaranarayanan@polymtl.ca).
- Open `Makefile`. 
- Enter the path of your armadillo-installation in the line defining the variable `ARMA`. Typically, the location would be like `/opt/armadillo-code`.
- Enter the path of your Gurobi-installation in the line defining the variable `GUR`. Typically, the location would be like `/opt/gurobi/gurobi811/<Your OS>`.
- Run `mkdir bin && mkdir dat` to create the folders where the executable file and other temporary files sit.
- Run `make compileEPEC` to compile. 
- Run `make` to run the code.  
