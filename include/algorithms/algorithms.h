#pragma once
#include "epecsolve.h"

namespace Algorithms {
///@brief The namespace Algorithms is responsible for the management of the
///algorithms that solve EPECs. Generally, the namespace is organized with
///multiple-level inheritances. The basic class is Algorithm, which implements
///some basic capabilities that all algorithms are sharing. Then, PolyBase
///managed the algorithm that either inner-approximate or full-enumerate the
///feasible region of each EPEC's player (3rd level inheritors: e.g.,
///Algorithm->PolyBase->FullEnumeration). The OuterApproximation class (2nd
///level inheritance) manages the outer approximation.

// the class generic stores some common information for algorithms
class Algorithm {
protected:
  GRBEnv *Env;
  Game::EPEC *EPECObject;
  virtual void postSolving() = 0;

public:
  virtual void solve()  = 0;
  virtual bool isSolved(double tol=-1) const = 0;
  virtual bool isPureStrategy(double tol=-1) const = 0;
};
// Second level inheritor for polyhedral inner approximations or full
// enumeration
class PolyBase;
// The following algorithms are children of polybase
class FullEnumeration;
class InnerApproximation;
class CombinatorialPNE;

// Then, second level inheritor for the outer approximation
class OuterApproximation;

} // namespace Algorithms