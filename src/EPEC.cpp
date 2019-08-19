#include<iostream>
#include<cstdlib>
#include<iterator>
#include<ctime>
#include"models.h"
#include<gurobi_c++.h>
#include<boost/program_options.hpp>
#include<boost/log/trivial.hpp>
#include<boost/log/core.hpp>
#include<boost/log/expressions.hpp>

using namespace std;
namespace logging = boost::log;
using namespace boost::program_options;
namespace po = boost::program_options;

int main(int argc, char **argv) {
    string resFile, instanceFile = "", logFile;
    int writeLevel, nThreads, verbosity;
    double timeLimit;

    po::options_description desc("EPEC: Allowed options");
    desc.add_options()
            ("help,h", "Shows this help message")
            ("version,v", "Shows EPEC version")
            ("input,i", po::value<string>(&instanceFile),
             "Sets the input path/filename of the instance file (.json appended automatically)")
            ("solution,s", po::value<string>(&resFile)->default_value("dat/Solution"),
             "Sets the output path/filename of the solution file (.json appended automatically)")
            ("log,l", po::value<string>(&logFile)->default_value("dat/Results.csv"),
             "Sets the output path/filename of the log file")
            ("timelimit,tl", po::value<double>(&timeLimit)->default_value(-1.0),
             "Sets the timelimit for solving the Nash Equilibrium model")
            ("writelevel,w", po::value<int>(&writeLevel)->default_value(0),
             "Sets the writeLevel param. 0: only Json. 1: only human-readable. 2: both")
            ("message,m", po::value<int>(&verbosity)->default_value(0),
             "Sets the verbosity level for info and warning messages. 0: warning and critical. 1: info. 2: trace")
            ("threads,t", po::value<int>(&nThreads)->default_value(1),
             "Sets the number of Threads for Gurobi. (int): number of threads. 0: auto (number of processors)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc;
        return EXIT_SUCCESS;
    }
    if (instanceFile == "") {
        cout << "-i [--input] option missing" << endl;
        return EXIT_SUCCESS;
    }
    if (verbosity == 0)
        logging::core::get()->set_filter(logging::trivial::severity > logging::trivial::info);
    else if (verbosity == 1)
        logging::core::get()->set_filter(logging::trivial::severity >= logging::trivial::info);
    else if (verbosity == 2) {
        logging::core::get()->set_filter(logging::trivial::severity >= logging::trivial::trace);
        arma::arma_version ver;
        int major, minor, technical;
        GRBversion(&major, &minor, &technical);
        BOOST_LOG_TRIVIAL(trace) << "Dependencies:\n\tARMAdillo: " << ver.as_string();
        BOOST_LOG_TRIVIAL(trace) << "\tGurobi: " << to_string(major)<<"."<<to_string(minor);
        BOOST_LOG_TRIVIAL(trace) << "\tBoost: " << to_string(BOOST_VERSION / 100000)<<"."<<to_string(BOOST_VERSION / 100 % 1000);
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
    /*char envThreads[(int) ceil((nThreads + 1) / 10)];
    strcpy(envThreads, to_string(nThreads).c_str());
    setenv("OPENBLAS_NUM_THREADS", envThreads, true);
     */
    Models::EPEC epec(&env);
    epec.timeLimit = timeLimit;
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
    if (stat.status == 1) epec.writeSolution(writeLevel, resFile);
    ifstream existCheck(logFile);
    std::ofstream results(logFile, ios::app);
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

    return EXIT_SUCCESS;
} 

