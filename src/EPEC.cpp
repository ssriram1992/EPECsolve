#include<iostream>
#include<memory>
#include<exception>
#include"models.h"
#include<gurobi_c++.h>
#include<armadillo>
#include<iomanip>

using namespace std;

static void show_usage(std::string name) {
    cerr << "Usage: " << name << " InstanceFile" << endl
         << "InstanceFile:\t\tThe path and file name of the JSON instance. **default: dat/Instance (.json automatically added)**\n\n"
         << "Options:\n"
         << "\t-h\t\t\tShow this help message\n"
         << "\t-v\t\t\tShow the version of EPEC\n"
         << "\t-r\t\t\tDictates the path and file name of the solution file. *default: dat/Solution (.json automatically added)**\n"
         // << "\t-l\t\t\tDictates the 'loquacity' of EPEC. Default: 0 (non-verbose); 1 (verbose)\n"
         << "\t-s\t\t\tDictates the writeLevel for EPEC solution. Default: 0 (only JSON); 1 (only Human Readable); 2 (both)\n"
         << endl;
}

int main(int argc, char *argv[]) {
    /**
    * @brief Pushes an instance from a file to the EPEC code
    * @p arg1 contains the
    */
    if (argc < 2) {
        show_usage(argv[0]);
        return 1;
    }
    string resFile = "dat/Solution";
    string instanceFile = "dat/Instance";
    int writeLevel = 0;


    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-h")) {
            show_usage(argv[0]);
            return 0;
        } else if ((arg == "-v")) {
            cout << "EPEC v" << to_string(EPECVERSION) << endl;
            return 0;
        } else if ((arg == "-r")) {
            if (i + 1 < argc) {
                resFile = argv[i++];
            } else {
                cerr << "-r option requires one argument." << endl;
                return 1;
            }
        } else if ((arg == "-s")) {
            if (i + 1 < argc) {
                writeLevel = strtol(argv[i++], NULL, 10);
            } else {
                cerr << "-s option requires one argument." << endl;
                return 1;
            }
        } else {
            instanceFile = argv[i];
        }
    }
    Models::EPECInstance Instance = Models::readInstance(instanceFile);
    if (Instance.Countries.size() < 1) {
        cerr << "Error: instance is empty" << endl;
        return 1;
    }
    GRBEnv env = GRBEnv();
    Models::EPEC epec(&env);
    for (int j = 0; j < Instance.Countries.size(); ++j)
        epec.addCountry(Instance.Countries.at(j));
    epec.addTranspCosts(Instance.TransportationCosts);
    epec.finalize();
    epec.make_country_QP();
    epec.findNashEq();
    epec.writeSolution(writeLevel, resFile);
    return 0;
} 

