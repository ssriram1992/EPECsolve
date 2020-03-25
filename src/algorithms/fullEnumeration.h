#pragma once
#include "algorithms/algorithms.h"
#include <armadillo>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>
#include <set>
#include <string>

namespace Algorithms {

///@brief This class is responsible for the fully enumerative algorithm
class fullEnumeration : public PolyBase {
public:
  fullEnumeration(GRBEnv *env, EPEC *EPECObject): PolyBase(env, EPECObject){};
  void solve();
};
} // namespace Algorithms