#pragma once

#include "lcp.h"
#include <armadillo>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>
#include <set>
#include <string>
using namespace std;

namespace Game {
class OuterLCP : public LCP {
  // using LCP::LCP;
  /**
   * @brief Inheritor Class to handle the outer approximation for the LCP class
   */
public:
  OuterLCP(GRBEnv *env, const NashGame &N) : LCP(env, N) {
    this->Ai = std::unique_ptr<spmat_Vec>(new spmat_Vec());
    this->bi = std::unique_ptr<vec_Vec>(new vec_Vec());
    this->clearApproximation();
  };
  void clearApproximation() {
    this->Ai->clear();
    this->bi->clear();
    this->Approximation.clear();
  }
  bool checkComponentFeas(const std::vector<short int> &encoding);
  void makeQP(Game::QP_Objective &QP_obj, Game::QP_Param &QP) override;
  void outerApproximate(std::vector<bool> encoding);
  bool addComponent(vector<short int> encoding, bool checkFeas,
                    bool custom = false, spmat_Vec *custAi = {},
                    vec_Vec *custbi = {});

private:
  std::unique_ptr<spmat_Vec>
      Ai; ///< Vector to contain the LHS of inner approx polyhedra
  std::unique_ptr<vec_Vec>
      bi; ///< Vector to contain the RHS of inner approx polyhedra
  std::set<unsigned long int> Approximation =
      {}; ///< Decimal encoding of polyhedra that have been enumerated
  std::set<unsigned long int> FeasibleComponents =
      {}; ///< Decimal encoding of polyhedra that have been enumerated
  std::set<unsigned long int> InfeasibleComponents =
      {}; ///< Decimal encoding of polyhedra known to be infeasible
  bool isParent(const vector<short> &father, const vector<short> &child);
  void addChildComponents(const vector<short> encoding);
  unsigned int convexHull(arma::sp_mat &A, arma::vec &b);
};
} // namespace Game