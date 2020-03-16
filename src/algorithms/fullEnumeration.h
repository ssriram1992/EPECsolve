#pragma once

#include "lcptolp.h"
#include <armadillo>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>
#include <set>
#include <string>

namespace Algorithms {

///@brief This class is responsible for the fully enumerative algorithm
class fullEnumeration {
private:
  GRBEnv *env;      ///< Stores the pointer to the Gurobi Environment
  EPEC *EPECObject; ///< Stores the pointer to the calling EPEC object

public:
  friend class EPEC;
  fullEnumeration(GRBEnv *env, EPEC *EpecObj)
      : env{env},
        EPECObject{EpecObj} {}; ///< Constructor requires a pointer to the Gurobi
                                ///< Environment and the calling EPEC object
  void solve();

};
} // namespace Game