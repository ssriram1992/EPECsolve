#include "combinatorialPNE.h"

#include <boost/log/trivial.hpp>
#include <chrono>
#include <gurobi_c++.h>
#include <set>

using namespace std;
using namespace Algorithms;

void Algorithms::combinatorialPNE::solve(
    const std::vector<std::set<unsigned long int>> &excludeList) {
  /** @brief Solve the referenced EPEC instance with the combinatorial
   *pure-equilibrium algorithm
   * @p excludelist contains the set of excluded polyhedra combinations.
   */
  if (this->EPECObject->Stats.AlgorithmParam.timeLimit > 0) {
    // Checking the function hasn't been called from innerApproximation
    if (this->EPECObject->Stats.numIteration <= 0) {
      this->EPECObject->initTime = std::chrono::high_resolution_clock::now();
    }
  }
  std::vector<long int> start;
  for (int j = 0; j < this->EPECObject->nCountr; ++j)
    start.push_back(-1);
  this->combPNE(start, excludeList);
  if (this->EPECObject->Stats.status==EPECsolveStatus::unInitialized)
    this->EPECObject->Stats.status = EPECsolveStatus::nashEqNotFound;
  this->postSolving();
  return;
}

void Algorithms::combinatorialPNE::combPNE(
    const std::vector<long int> combination,
    const std::vector<std::set<unsigned long int>> &excludeList) {
  /** @brief Starting from @p combination, the methods builds the recursion to
   * generate the subproblems associated with all the existing combinations of
   * polyhedra. Then, it solves each subproblem, and if a solution is found, it
   * terminates and  stores the solution  into the referenced EPEC object. @p
   * excludeList contains the excluded combinations of polyhedra.
   */
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
         j < this->poly_LCP.at(i)->getNumTheoreticalPoly();
         ++j) {
      if (this->poly_LCP.at(i)->checkPolyFeas(j)) {
        childCombination.at(i) = j;
        this->combPNE(childCombination, excludeList);
      }
    }
  } else {
    // Combination is filled and ready!
    // Check that this combination is not in the excuded list
    BOOST_LOG_TRIVIAL(trace)
        << "Algorithms::combinatorialPNE::combPNE: considering a FULL combination";
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
          << "Algorithms::combinatorialPNE::combPNE: considering a "
             "FEASIBLE combination of polyhedra.";
      for (int j = 0; j < this->EPECObject->nCountr; ++j) {
        this->poly_LCP.at(j)->clearPolyhedra();
        this->poly_LCP.at(j)->addThePoly(
            childCombination.at(j));
      }
      this->EPECObject->make_country_QP();
      bool res = false;
      if (this->EPECObject->Stats.AlgorithmParam.timeLimit > 0) {
        const std::chrono::duration<double> timeElapsed =
            std::chrono::high_resolution_clock::now() -
            this->EPECObject->initTime;
        const double timeRemaining =
            this->EPECObject->Stats.AlgorithmParam.timeLimit -
            timeElapsed.count();
        res = this->EPECObject->computeNashEq(false, timeRemaining, true);
      } else
        res = this->EPECObject->computeNashEq(false, -1.0, true);

      if (res) {
        if (this->EPECObject->isSolved()) {
          // Check that the equilibrium is a pure strategy
          if ((this->EPECObject->isPureStrategy())) {
            BOOST_LOG_TRIVIAL(info)
                << "Algorithms::combinatorialPNE::combPNE: found a pure strategy.";
            this->EPECObject->Stats.status = Game::EPECsolveStatus::nashEqFound;
            this->EPECObject->Stats.pureNE = true;
            return;
          }
        }
      }
    } else {
      BOOST_LOG_TRIVIAL(trace)
          << "Algorithms::combinatorialPNE::combPNE: configuration pruned.";
      return;
    }
  }
}