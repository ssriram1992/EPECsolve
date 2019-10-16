#include "models.h"
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdlib>
#include <gurobi_c++.h>
#include <iostream>
#include <iterator>

using namespace std;
namespace logging = boost::log;
using namespace boost::program_options;
namespace po = boost::program_options;

int main(int argc, char **argv) {
  string resFile, instanceFile = "", logFile;
  int writeLevel, nThreads, verbosity, bigM, algorithm, aggressiveness, add{0};
  double timeLimit, boundBigM;
  bool bound;

  po::options_description desc("EPEC: Allowed options");
  desc.add_options()("help,h", "Shows this help message")("version,v",
                                                          "Shows EPEC version")(
      "input,i", po::value<string>(&instanceFile),
      "Sets the input path/filename of the instance file (.json appended "
      "automatically)")(
      "algorithm,a", po::value<int>(&algorithm),
      "Sets the algorithm. 0: fullEnumeration, 1:innerApproximation")(
      "solution,s", po::value<string>(&resFile)->default_value("dat/Solution"),
      "Sets the output path/filename of the solution file (.json appended "
      "automatically)")(
      "log,l", po::value<string>(&logFile)->default_value("dat/Results.csv"),
      "Sets the output path/filename of the log file")(
      "timelimit,tl", po::value<double>(&timeLimit)->default_value(-1.0),
      "Sets the timelimit for solving the Nash Equilibrium model")(
      "writelevel,w", po::value<int>(&writeLevel)->default_value(0),
      "Sets the writeLevel param. 0: only Json. 1: only human-readable. 2: "
      "both")("message,m", po::value<int>(&verbosity)->default_value(0),
              "Sets the verbosity level for info and warning messages. 0: "
              "warning and critical. 1: info. 2: debug. 3: trace")(
      "bigm,b", po::value<int>(&bigM)->default_value(0),
      "Replaces indicator constraints with bigM.")(
      "threads,t", po::value<int>(&nThreads)->default_value(1),
      "Sets the number of Threads for Gurobi. (int): number of threads. 0: "
      "auto (number of processors)")(
      "aggr,ag", po::value<int>(&aggressiveness)->default_value(1),
      "Sets the aggressiveness for the innerApproximation, namely the number "
      "of random polyhedra added if no deviation is found. (int)")(
      "b,bound", po::value<bool>(&bound)->default_value(false),
      "Decides whether QP param should be bounded or not.")(
      "boundBigM", po::value<double>(&boundBigM)->default_value(1e5),
      "Set the bounding bigM related to the parameter --bound")(
      "add,ad", po::value<int>(&add)->default_value(0),
      "Sets the EPECAddPolyMethod for the innerApproximation. 0: sequential. "
      "1: reverse_sequential. 2:random.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc;
    return EXIT_SUCCESS;
  }
  if (instanceFile == "") {
    cout << "-i [--input] option missing.\n Use with --help for help on list "
            "of arguments\n";
    return EXIT_SUCCESS;
  }
  switch (verbosity) {
  case 0:
    logging::core::get()->set_filter(logging::trivial::severity >
                                     logging::trivial::info);
    break;
  case 1:
    logging::core::get()->set_filter(logging::trivial::severity >=
                                     logging::trivial::info);
    break;
  case 2:
    logging::core::get()->set_filter(logging::trivial::severity >=
                                     logging::trivial::debug);
    break;
  case 3:
    logging::core::get()->set_filter(logging::trivial::severity >=
                                     logging::trivial::trace);
    break;
  default:
    BOOST_LOG_TRIVIAL(warning)
        << "Invalid option for --message (-m). Setting default value: 0";
    verbosity = 0;
    logging::core::get()->set_filter(logging::trivial::severity >
                                     logging::trivial::info);
    break;
  }
  if (verbosity >= 2) {
    arma::arma_version ver;
    int major, minor, technical;
    GRBversion(&major, &minor, &technical);
    BOOST_LOG_TRIVIAL(info)
        << "Dependencies:\n\tARMAdillo: " << ver.as_string();
    BOOST_LOG_TRIVIAL(info)
        << "\tGurobi: " << to_string(major) << "." << to_string(minor);
    BOOST_LOG_TRIVIAL(info) << "\tBoost: " << to_string(BOOST_VERSION / 100000)
                            << "." << to_string(BOOST_VERSION / 100 % 1000);
  }
  // --------------------------------
  // LOADING INSTANCE
  // --------------------------------
  Models::EPECInstance Instance(instanceFile);
  if (Instance.Countries.empty()) {
    cerr << "Error: instance is empty\n";
    return 1;
  }

  // --------------------------------
  // TEST STARTS
  // --------------------------------
  auto time_start = std::chrono::high_resolution_clock::now();
  GRBEnv env = GRBEnv();

  // OPTIONS
  //------------
  Models::EPEC epec(&env);
  // Indicator constraints
  if (bigM == 1)
    epec.setIndicators(false);
  // Num Threads
  if (nThreads != 0)
    epec.setNumThreads(nThreads);
  // timeLimit
  epec.setTimeLimit(timeLimit);
  // bound QPs
  if (bound) {
    epec.setBoundQPs(true);
    epec.setBoundBigM(boundBigM);
  }

  // Algorithm

  switch (algorithm) {
  case 1: {
    epec.setAlgorithm(Game::EPECalgorithm::innerApproximation);
    if (aggressiveness != 1)
      epec.setAggressiveness(aggressiveness);
    switch (add) {
    case 1:
      epec.setAddPolyMethod(EPECAddPolyMethod::reverse_sequential);
      break;
    case 2:
      epec.setAddPolyMethod(EPECAddPolyMethod::random);
      break;
    default:
      epec.setAddPolyMethod(EPECAddPolyMethod::sequential);
    }

    break;
  }
  default:
    epec.setAlgorithm(Game::EPECalgorithm::fullEnumeration);
  }
  //------------

  for (unsigned int j = 0; j < Instance.Countries.size(); ++j)
    epec.addCountry(Instance.Countries.at(j));
  epec.addTranspCosts(Instance.TransportationCosts);
  epec.finalize();
  // epec.make_country_QP();
  try {
    epec.findNashEq();
  } catch (string &s) {
    std::cerr << "Error while finding Nash equilibrium: " << s << '\n';
    ;
  } catch (exception &e) {
    std::cerr << "Error while finding Nash equilibrium: " << e.what() << '\n';
    ;
  }
  auto time_stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_diff = time_stop - time_start;
  double WallClockTime = time_diff.count();
  int realThreads = nThreads > 0 ? env.get(GRB_IntParam_Threads) : nThreads;

  // --------------------------------
  // WRITING STATISTICS AND SOLUTION
  // --------------------------------
  Game::EPECStatistics stat = epec.getStatistics();
  if (stat.status == Game::EPECsolveStatus::nashEqFound)
    epec.writeSolution(writeLevel, resFile);
  ifstream existCheck(logFile);
  std::ofstream results(logFile, ios::app);

  if (!existCheck.good()) {
    results
        << "Instance;Algorithm;Countries;Followers;Status;numFeasiblePolyhedra;"
           "numVar;numConstraints;numNonZero;ClockTime"
           "(s);Threads;Indicators;numInnerIterations;lostIntermediateEq;"
           "Aggressiveness;"
           "AddPolyMethod;numericalIssuesEncountered;bound;boundBigM\n";
  }
  existCheck.close();

  stringstream PolyT;
  copy(stat.feasiblePolyhedra.begin(), stat.feasiblePolyhedra.end(),
       ostream_iterator<int>(PolyT, " "));

  results << instanceFile << ";" << to_string(epec.getAlgorithm()) << ";"
          << Instance.Countries.size() << ";[";
  for (auto &Countrie : Instance.Countries)
    results << " " << Countrie.n_followers;

  results << " ];" << to_string(stat.status) << ";[ " << PolyT.str() << "];"
          << stat.numVar << ";" << stat.numConstraints << ";" << stat.numNonZero
          << ";" << WallClockTime << ";" << realThreads << ";"
          << to_string(epec.getIndicators());
  if (epec.getAlgorithm() == Game::EPECalgorithm::innerApproximation) {
    results << ";" << epec.getStatistics().numIteration << ";"
            << epec.getStatistics().lostIntermediateEq << ";"
            << epec.getAggressiveness() << ";"
            << to_string(epec.getAddPolyMethod()) << ";"
            << epec.getStatistics().numericalIssuesEncountered << ";"
            << to_string(epec.getBoundQPs()) << ";" << epec.getBoundBigM();
  } else {
    results << ";-;-;-;-;-;-;-";
  }
  results << "\n";
  results.close();

  return EXIT_SUCCESS;
}
