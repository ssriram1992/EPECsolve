#pragma once
#include "epecsolve.h"

namespace Algorithms {
class PolyBase {
  /*
   *  @brief This is the abstract class of Algorithms for full enumeration,
   * inner approximation, and Combinatorial PNE. It provides a constructor where
   * the Gurobi environment and the EPEC are passed. This is an abstract class.
   */
protected:
  std::vector<std::shared_ptr<Game::PolyLCP>> PolyLCP{};
  GRBEnv *Env;
  Game::EPEC *EPECObject;

  void postSolving() {
    /**
     * Perform postSolving operations.
     * For instance, it updates the statistics associated with the feasible
     * polyhedra. The responsability for calling this method is left to the
     * inheritor
     */
    for (unsigned int i = 0; i < this->EPECObject->NumPlayers; i++)
      this->EPECObject->Stats.FeasiblePolyhedra.at(i) =
          this->PolyLCP.at(i)->getFeasiblePolyhedra();
  }

public:
  PolyBase(GRBEnv *env, Game::EPEC *EPECObject) {
    /*
     *  The method will reassign the LCP fields in the EPEC object to new
     * PolyLCP objects
     */
    this->EPECObject = EPECObject;
    this->Env = env;
    this->EPECObject->Stats.AlgorithmParam.PolyLcp = true;
    this->PolyLCP =
        std::vector<std::shared_ptr<Game::PolyLCP>>(EPECObject->NumPlayers);
    for (unsigned int i = 0; i < EPECObject->NumPlayers; i++) {
      this->PolyLCP.at(i) =
          std::shared_ptr<Game::PolyLCP>(new class Game::PolyLCP(
              this->Env, *EPECObject->PlayersLowerLevels.at(i).get()));
      EPECObject->PlayersLCP.at(i) = this->PolyLCP.at(i);
    }
  }
  virtual void solve() = 0;
};
} // namespace Algorithms