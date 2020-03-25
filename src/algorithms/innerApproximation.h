#pragma once
#include "algorithms/algorithms.h"
#include <armadillo>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>
#include <set>
#include <string>

namespace Algorithms {

///@brief This class is responsible for the inner Approximation
class innerApproximation : public PolyBase {

public:
  innerApproximation(GRBEnv *env, EPEC *EPECObject)
      : PolyBase(env, EPECObject){};
  void solve() override;

private:
  bool addRandomPoly2All(unsigned int aggressiveLevel = 1,
                         bool stopOnSingleInfeasibility = false);
  bool getAllDevns(std::vector<arma::vec> &devns, const arma::vec &guessSol,
                   const std::vector<arma::vec> &prevDev = {}) const;
  unsigned int addDeviatedPolyhedron(const std::vector<arma::vec> &devns,
                                     bool &infeasCheck) const;
};
} // namespace Algorithms