#include "algorithms/outerapproximation.h"

#include <boost/log/trivial.hpp>
#include <chrono>
#include <gurobi_c++.h>
#include <set>
#include <string>

using namespace std;
void Algorithms::OuterApproximation::solve() {
  /**
   * Given the referenced EPEC instance, this method solves it through the outer
   * approximation Algorithm.
   */
  // Set the initial point for all countries as 0 and solve the respective LCPs?
  this->EPECObject->SolutionX.zeros(this->EPECObject->NumVariables);
  bool solved = {false};
  bool addRand{false};
  bool infeasCheck{false};

  this->EPECObject->Stats.NumIterations = 0;
  if (this->EPECObject->Stats.AlgorithmParam.TimeLimit > 0)
    this->EPECObject->InitTime = std::chrono::high_resolution_clock::now();

  // Initialize Trees
  this->Trees = std::vector<Tree *>(this->EPECObject->NumPlayers, 0);
  for (unsigned int i = 0; i < this->EPECObject->NumPlayers; i++)
    Trees.at(i) = new Tree(this->EPECObject->PlayersLCP.at(i)->getNumRows());

  auto leaves = Trees.at(0)->branch(1, Trees.at(0)->getRoot());

  while (!solved) {
    ++this->EPECObject->Stats.NumIterations;
    BOOST_LOG_TRIVIAL(info)
        << "Algorithms::OuterApproximation::solve: Iteration "
        << to_string(this->EPECObject->Stats.NumIterations);
  }
}