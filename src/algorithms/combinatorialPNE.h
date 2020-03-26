#pragma once
#include "algorithms/algorithms.h"

namespace Algorithms {

///@brief This class is responsible for the combinatorial pure-nash Equilibrium
class combinatorialPNE : public PolyBase {
public:
  combinatorialPNE(GRBEnv *env, EPEC *EPECObject, bool poly = true): PolyBase(env, EPECObject){};;
  void solve() override {
    this->solve(std::vector<std::set<unsigned long int>>{});
  }
  void solve(const std::vector<std::set<unsigned long int>> &excludeList);

private:
  // Making the method private
  void combPNE(const std::vector<long int> combination,
               const std::vector<std::set<unsigned long int>> &excludeList);
};
} // namespace Algorithms