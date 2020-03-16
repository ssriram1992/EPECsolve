#include "combinatorialPNE.h"
#include "games.h"
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


void Game::combinatorialPNE::solve(
    const std::vector<long int> combination,
    const std::vector<std::set<unsigned long int>> &excludeList) {

  if (this->EPECObject->Stats.AlgorithmParam.timeLimit > 0) {
    // Checking the function hasn't been called from innerApproximation
    if (this->EPECObject->Stats.numIteration <= 0) {
      this->EPECObject->initTime = std::chrono::high_resolution_clock::now();
    }
  }
  if (combination.empty()) {
    std::vector<long int> start;
    for (int j = 0; j < this->EPECObject->nCountr; ++j)
      start.push_back(-1);
    this->recursion(start, excludeList);
  } else
    this->recursion(combination, excludeList);

  return;
}


void Game::combinatorialPNE::recursion(
    const std::vector<long int> combination,
    const std::vector<std::set<unsigned long int>> &excludeList) {

  if ((this->EPECObject->Stats.status == EPECsolveStatus::nashEqFound &&
       this->EPECObject->Stats.pureNE == true) ||
      this->EPECObject->Stats.status == EPECsolveStatus::timeLimit)
    return;

  if (this->EPECObject->Stats.AlgorithmParam.timeLimit > 0) {
    const std::chrono::duration<double> timeElapsed =
        std::chrono::high_resolution_clock::now() - this->EPECObject->initTime;
    const double timeRemaining =
        this->EPECObject->Stats.AlgorithmParam.timeLimit - timeElapsed.count();
    if (timeRemaining <= 0) {
      this->EPECObject->Stats.status = Game::EPECsolveStatus::timeLimit;
      return;
    }
  }

  std::vector<long int> childCombination(combination);
  bool found{false};
  unsigned int i{0};
  for (i = 0; i < this->EPECObject->nCountr; i++) {
    if (childCombination.at(i) == -1) {
      found = true;
      break;
    }
  }
  if (found) {
    for (unsigned int j = 0;
         j < this->EPECObject->countries_LCP.at(i)->getNumTheoreticalPoly(); ++j) {
      if (this->EPECObject->countries_LCP.at(i)->checkPolyFeas(j)) {
        childCombination.at(i) = j;
        this->recursion(childCombination, excludeList);
      }
    }
  } else {
    // Combination is filled and ready!
    // Check that this combination is not in the excuded list
    BOOST_LOG_TRIVIAL(trace)
      << "Game::EPEC::combinatorial_pure_NE: considering a FULL combination";
    bool excluded = false;
    if (!excludeList.empty()) {
      excluded = true;
      for (unsigned int j = 0; j < this->EPECObject->nCountr; ++j) {
        if (excludeList.at(j).find(childCombination.at(j)) ==
            excludeList.at(j).end()) {
          excluded = false;
        }
      }
    }

    if (!excluded) {
      BOOST_LOG_TRIVIAL(trace)
        << "Game::EPEC::combinatorial_pure_NE: considering a "
           "FEASIBLE combination of polyhedra.";
      for (int j = 0; j < this->EPECObject->nCountr; ++j) {
        this->EPECObject->countries_LCP.at(j)->clearPolyhedra();
        this->EPECObject->countries_LCP.at(j)->addThePoly(childCombination.at(j));
      }
      this->EPECObject->make_country_QP();
      bool res = false;
      if (this->EPECObject->Stats.AlgorithmParam.timeLimit > 0) {
        const std::chrono::duration<double> timeElapsed =
            std::chrono::high_resolution_clock::now() - this->EPECObject->initTime;
        const double timeRemaining =
            this->EPECObject->Stats.AlgorithmParam.timeLimit - timeElapsed.count();
        res = this->EPECObject->computeNashEq(false, timeRemaining, true);
      } else
        res = this->EPECObject->computeNashEq(false, -1.0, true);

      if (res) {
        if (this->EPECObject->isSolved()) {
          // Check that the equilibrium is a pure strategy
          if ((this->EPECObject->isPureStrategy())) {
            BOOST_LOG_TRIVIAL(info)
              << "Game::EPEC::combinatorial_pure_NE: found a pure strategy.";
            this->EPECObject->Stats.status = Game::EPECsolveStatus::nashEqFound;
            this->EPECObject->Stats.pureNE = true;
            return;
          }
        }
      }
    } else {
      BOOST_LOG_TRIVIAL(trace)
        << "Game::EPEC::combinatorial_pure_NE: configuration pruned.";
      return;
    }
  }
}