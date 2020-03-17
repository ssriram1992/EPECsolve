#pragma once

#include "../src/games.h"
#include <armadillo>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>
#include <set>
#include <string>

namespace Algorithms {

///@brief This class is responsible for the combinatorial pure-nash Equilibrium
class combinatorialPNE {
private:
  GRBEnv *env;      ///< Stores the pointer to the Gurobi Environment
  EPEC *EPECObject; ///< Stores the pointer to the calling EPEC object

public:
  friend class Game::EPEC;
  combinatorialPNE(GRBEnv *env, EPEC *EpecObj)
      : env{env},
        EPECObject{EpecObj} {}; ///< Constructor requires a pointer to the Gurobi
                                ///< Environment and the calling EPEC object
  void solve(const std::vector<std::set<unsigned long int>> &excludeList = {});

private:
  void combPNE(const std::vector<long int> combination,
               const std::vector<std::set<unsigned long int>> &excludeList);
};
} // namespace Algorithms