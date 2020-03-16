#pragma once

#include "lcptolp.h"
#include <armadillo>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>
#include <set>
#include <string>

namespace Game {

class combinatorialPNE {
private:
  GRBEnv *env;
  EPEC *EPECObject;

public:
  friend class EPEC;
  combinatorialPNE(GRBEnv *env, EPEC *EpecObj)
      : env{env}, EPECObject{EpecObj} {};
  void
  solve(const std::vector<long int> combination = {},
        const std::vector<std::set<unsigned long int>> &excludeList = {});
  void recursion(const std::vector<long int> combination,
                             const std::vector<std::set<unsigned long int>> &excludeList);
};
} // namespace Game