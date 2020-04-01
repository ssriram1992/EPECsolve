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
class InnerApproximation : public PolyBase {

public:
  InnerApproximation(GRBEnv *env, EPEC *EPECObject)
      : PolyBase(env, EPECObject){};
  void solve() override;

private:
  void start();
  bool addRandomPoly2All(unsigned int aggressiveLevel = 1,
                         bool stopOnSingleInfeasibility = false);
  bool getAllDeviations(std::vector<arma::vec> &deviations,
                        const arma::vec &guessSol,
                        const std::vector<arma::vec> &prevDev = {}) const;
  unsigned int addDeviatedPolyhedron(const std::vector<arma::vec> &deviations,
                                     bool &infeasCheck) const;
};
} // namespace Algorithms