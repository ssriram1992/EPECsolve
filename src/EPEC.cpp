#include<iostream>
#include<cstdlib>
#include<iterator>
#include<ctime>
#include<math.h>
#include"models.h"
#include<gurobi_c++.h>

using namespace std;

static void show_usage(std::string name) {
    cerr << "Usage: " << name << " InstanceFile" << endl
         << "InstanceFile:\t\tThe path and file name of the JSON instance. **default: dat/Instance (.json automatically added)**\n\n"
         << "Options:\n"
         << "\t-h\t\t\tShow this help message\n"
         << "\t-v\t\t\tShow the version of EPEC\n"
         << "\t-r\t\t\tDictates the path and file name of the solution file. *default: dat/Solution (.json automatically added)**\n"
         << "\t-rf\t\t\tDictates the path and file name of the general results file. *default: dat/results.csv **\n"
         // << "\t-l\t\t\tDictates the 'loquacity' of EPEC. Default: 0 (non-verbose); 1 (verbose)\n"
         << "\t-s\t\t\tDictates the writeLevel for EPEC solution. Default: 0 (only JSON); 1 (only Human Readable); 2 (both)\n"
         << "\t-t\t\t\tNumber of threads for Gurobi and OPEN_BLAS. Default: 1; int (number of cores); 0 (auto)\n"
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
    string resultsFile = "dat/results.csv";
    int writeLevel = 0;
    int nThreads = 1;


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
                resFile = argv[++i];
            } else {
                cerr << "-r option requires one argument." << endl;
                return 1;
            }
        } else if ((arg == "-rf")) {
            if (i + 1 < argc) {
                resultsFile = argv[++i];
            } else {
                cerr << "-rf option requires one argument." << endl;
                return 1;
            }
        } else if ((arg == "-s")) {
            if (i + 1 < argc) {
                writeLevel = strtol(argv[++i], NULL, 10);
            } else {
                cerr << "-s option requires one argument." << endl;
                return 1;
            }
        } else if ((arg == "-t")) {
            if (i + 1 < argc) {
                nThreads = strtol(argv[++i], NULL, 10);
            } else {
                cerr << "-t option requires one argument." << endl;
                return 1;
            }
        } else {
            instanceFile = argv[i];
        }
    }

    // --------------------------------
    // LOADING INSTANCE
    // --------------------------------
    Models::EPECInstance Instance(instanceFile);
    if (Instance.Countries.empty()) {
        cerr << "Error: instance is empty" << endl;
        return 1;
    }

    // --------------------------------
    // TEST STARTS
    // --------------------------------
    clock_t time_start = clock();
    GRBEnv env = GRBEnv();
    env.set(GRB_IntParam_Threads, nThreads);
    char envThreads[(int) ceil((nThreads + 1) / 10)];
    strcpy(envThreads, to_string(nThreads).c_str());
    setenv("OPENBLAS_NUM_THREADS", envThreads, true);
    Models::EPEC epec(&env);
    for (int j = 0; j < Instance.Countries.size(); ++j)
        epec.addCountry(Instance.Countries.at(j));
    epec.addTranspCosts(Instance.TransportationCosts);
    epec.finalize();
    epec.make_country_QP();
    try {
        epec.findNashEq();
    }
    catch (...) {}
    clock_t time_stop = clock();
    double CPUTime = 1000.0 * (time_stop - time_start) / CLOCKS_PER_SEC;

    // --------------------------------
    // WRITING STATISTICS AND SOLUTION
    // --------------------------------
    Models::EPECStatistics stat = epec.getStatistics();
    if (stat.status) epec.writeSolution(writeLevel, resFile);
    ifstream existCheck(resultsFile);
    std::ofstream results(resultsFile, ios::app);
    if (!existCheck.good()) {
        results
                << "Instance;Countries;Followers;Status;numFeasiblePolyhedra;numVar;numConstraints;numNonZero;CPUTime (ms)\n";
    }
    existCheck.close();
    stringstream PolyT;
    copy(stat.feasiblePolyhedra.begin(), stat.feasiblePolyhedra.end(), ostream_iterator<int>(PolyT, " "));

    results << instanceFile << ";" << to_string(Instance.Countries.size()) << ";[";
    for (auto &Countrie : Instance.Countries)
        results << " " << to_string(Countrie.n_followers);
    results << " ];" << to_string(stat.status) << ";[ " << PolyT.str() << "];" << to_string(stat.numVar) << ";"
            << to_string(stat.numConstraints)
            << ";" << to_string(stat.numNonZero) << ";" << to_string(CPUTime) << "\n";
    results.close();
    return 0;
} 

