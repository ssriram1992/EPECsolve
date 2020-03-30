#pragma once
#include "algorithms/algorithms.h"

namespace Algorithms {

///@brief This class is responsible for the fully enumerative algorithm
class fullEnumeration : public PolyBase {
public:
  fullEnumeration(GRBEnv *env, EPEC *EPECObject): PolyBase(env, EPECObject){};
  void solve() override;
};
} // namespace Algorithms