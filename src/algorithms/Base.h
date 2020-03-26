#pragma once
#include "epecsolve.h"

namespace Algorithms {
class PolyBase {
  /*
 *  @brief This is the abstract class of Algorithms for full enumeration, inner approximation, and combinatorial PNE.
   *  It provides a constructor where the Gurobi environment and the EPEC is passed. An abstract
 */
protected:
  std::vector<std::shared_ptr<Game::polyLCP>> poly_LCP{};
  GRBEnv *env;
  EPEC *EPECObject;

  void postSolving() {
    /**
     * Perform postSolving operations.
     * For instance, it updates the statistics associated with the feasible polyhedra.
     * The responsability for calling this method is left to the
     * inheritor
     */
    for (unsigned int i = 0; i < this->EPECObject->nCountr; i++)
      this->EPECObject->Stats.feasiblePolyhedra.at(i) =
          this->poly_LCP.at(i)->getFeasiblePolyhedra();
  }
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
  virtual void solve(){};
};
} // namespace Algorithms