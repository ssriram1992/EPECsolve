#pragma once

#include "lcptolp.h"
#include <armadillo>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>
#include <set>
#include <string>

namespace Game {

class innerApproximation {
private:
  GRBEnv *env;
  EPEC *EPECObject;

public:
  friend class EPEC;
  innerApproximation(GRBEnv *env, EPEC *EpecObj)
      : env{env}, EPECObject{EpecObj} {};
  void solve();
  bool addRandomPoly2All(unsigned int aggressiveLevel = 1,
                         bool stopOnSingleInfeasibility = false);
  bool getAllDevns(std::vector<arma::vec> &devns, const arma::vec &guessSol,
                   const std::vector<arma::vec> &prevDev = {}) const;
  unsigned int addDeviatedPolyhedron(const std::vector<arma::vec> &devns,
                                     bool &infeasCheck) const;
};
} // namespace Game