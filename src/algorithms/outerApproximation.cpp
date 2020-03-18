#include "outerApproximation.h"

#include <algorithm>
#include <armadillo>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>

using namespace std;
void Algorithms::outerApproximation::solve() {
  /**
   * Given the referenced EPEC instance, this method solves it through the outer
   * approximation algorithm.
   */
  // Set the initial point for all countries as 0 and solve the respective LCPs?
  this->EPECObject->sol_x.zeros(this->EPECObject->nVarinEPEC);
  bool solved = {false};
  bool addRand{false};
  bool infeasCheck{false};

  this->EPECObject->Stats.numIteration = 0;
  if (this->EPECObject->Stats.AlgorithmParam.timeLimit > 0)
    this->EPECObject->initTime = std::chrono::high_resolution_clock::now();

  // Initialize Trees
  this->Trees = std::vector<Tree *>(this->EPECObject->nCountr, 0);
  for (unsigned int i = 0; i < this->EPECObject->nCountr; i++)
    Trees.at(i) = new Tree(this->EPECObject->countries_LCP.at(i)->getNrow());

  auto leaves =  Trees.at(0)->branch(1,Trees.at(0)->getRoot());

  while (!solved) {
    ++this->EPECObject->Stats.numIteration;
    BOOST_LOG_TRIVIAL(info)
        << "Algorithms::outerApproximation::solve: Iteration "
        << to_string(this->EPECObject->Stats.numIteration);
  }

}