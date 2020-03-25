#pragma once
#include "epecsolve.h"

namespace Algorithms {
class PolyBase {
protected:
  std::vector<std::shared_ptr<Game::polyLCP>> poly_LCP{};
  GRBEnv *env;
  EPEC *EPECObject;

public:
  PolyBase(GRBEnv *env, EPEC *EPECObject) {
    /*
     *  The method will reassign the LCP fields in the EPEC object to new
     * PolyLCP objects
     */
    this->EPECObject = EPECObject;
    this->env = env;
    this->poly_LCP =
        std::vector<std::shared_ptr<Game::polyLCP>>(EPECObject->nCountr);
    for (unsigned int i = 0; i < EPECObject->nCountr; i++) {
      this->poly_LCP.at(i) = std::shared_ptr<Game::polyLCP>(
          new polyLCP(this->env, *EPECObject->countries_LL.at(i).get()));
      EPECObject->countries_LCP.at(i) = this->poly_LCP.at(i);
    }
  }
  void poly_updatePolyStats() {
    for (unsigned int i = 0; i < this->EPECObject->nCountr; i++)
      this->EPECObject->Stats.feasiblePolyhedra.at(i) =
          this->poly_LCP.at(i)->getFeasiblePolyhedra();
  }
  virtual void solve() = 0;
};
} // namespace Algorithms