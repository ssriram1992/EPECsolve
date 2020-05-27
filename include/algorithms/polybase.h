#pragma once
#include "epecsolve.h"
#include "algorithms.h"
#include <boost/log/trivial.hpp>

namespace Algorithms {
class PolyBase : public Algorithm {
  /*
   *  @brief This is the abstract class of Algorithms for full enumeration,
   * inner approximation, and Combinatorial PNE. It provides a constructor where
   * the Gurobi environment and the EPEC are passed. This is an abstract class.
   */
protected:
  std::vector<std::shared_ptr<Game::PolyLCP>> PolyLCP{};

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
    this->EPECObject->Stats.PureNashEquilibrium = this->isPureStrategy();
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
  bool isSolved(unsigned int *countryNumber, arma::vec *profitableDeviation,
                double tol=-1) const;
  bool isSolved(double tol=-1) const override;
  void makeThePureLCP(bool indicators);

  double getValLeadFollPoly(unsigned int i, unsigned int j, unsigned int k,
                            double tol = 1e-5) const;

  double getValLeadLeadPoly(unsigned int i, unsigned int j, unsigned int k,
                            double tol = 1e-5) const;

  double getValProbab(unsigned int i, unsigned int k) const;

  bool isPureStrategy(unsigned int i, double tol = 1e-5) const;

  bool isPureStrategy(double tol = 1e-5) const override;

  std::vector<unsigned int> mixedStrategyPoly(unsigned int i,
                                              double tol = 1e-5) const;
  unsigned int getPositionLeadFollPoly(unsigned int i, unsigned int j,
                                       unsigned int k) const;

  unsigned int getPositionLeadLeadPoly(unsigned int i, unsigned int j,
                                       unsigned int k) const;

  unsigned int getNumPolyLead(unsigned int i) const;

  unsigned int getPositionProbab(unsigned int i, unsigned int k) const;
};
} // namespace Algorithms