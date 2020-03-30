#pragma once

#include "LCP.h"
#include <armadillo>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>
#include <set>
#include <string>

namespace Game {
class polyLCP : public LCP {
  //using LCP::LCP;
  /**
   * @brief Inheritor Class to handle the polyhedral aspects of the LCP class,
   * and support algorithms.
   */

private:
  int polyCounter{0};
  unsigned int feasiblePolyhedra{0};
  unsigned int sequentialPolyCounter{0};
  int reverseSequentialPolyCounter{0};
  /// LCP feasible region is a union of polyhedra. Keeps track which of those
  /// inequalities are fixed to equality to get the individual polyhedra
  std::set<unsigned long int> AllPolyhedra =
      {}; ///< Decimal encoding of polyhedra that have been enumerated
  std::set<unsigned long int> feasiblePoly =
      {}; ///< Decimal encoding of polyhedra that have been enumerated
  std::set<unsigned long int> knownInfeas =
      {}; ///< Decimal encoding of polyhedra known to be infeasible
  unsigned long int maxTheoreticalPoly{0};
  std::unique_ptr<spmat_Vec>
      Ai; ///< Vector to contain the LHS of inner approx polyhedra
  std::unique_ptr<vec_Vec>
      bi; ///< Vector to contain the RHS of inner approx polyhedra
  void initializeNotProcessed() {
    const unsigned int nCompl = this->Compl.size();
    // 2^n - the number of polyhedra theoretically
    this->maxTheoreticalPoly = static_cast<unsigned long int>(pow(2, nCompl));
    sequentialPolyCounter = 0;
    reverseSequentialPolyCounter = this->maxTheoreticalPoly - 1;
  }
  bool FixToPoly(const std::vector<short int> Fix, bool checkFeas = false,
                 bool custom = false, spmat_Vec *custAi = {},
                 vec_Vec *custbi = {});
  polyLCP &FixToPolies(const std::vector<short int> Fix,
                        bool checkFeas = false, bool custom = false,
                        spmat_Vec *custAi = {}, vec_Vec *custbi = {});
  unsigned long int getNextPoly(Game::EPECAddPolyMethod method);

public:
  polyLCP(GRBEnv *env, const NashGame &N)  : LCP(env, N){
    this->Ai = std::unique_ptr<spmat_Vec>(new spmat_Vec());
    this->bi = std::unique_ptr<vec_Vec>(new vec_Vec());
    this->clearPolyhedra();
    this->initializeNotProcessed();
  };
  long int addPolyMethodSeed = {
      -1}; ///< Seeds the random generator for the random polyhedra selection.
  ///< Should be a positive value
  /* Convex hull computation */
  unsigned int ConvexHull(arma::sp_mat &A, arma::vec &b);
  unsigned int conv_Npoly() const;
  unsigned int conv_PolyPosition(const unsigned long int i) const;
  unsigned int conv_PolyWt(const unsigned long int i) const;

  std::set<unsigned long int> getAllPolyhedra() const {
    return this->AllPolyhedra;
  };
  unsigned long int getNumTheoreticalPoly() const noexcept {
    return this->maxTheoreticalPoly;
  }
  std::set<std::vector<short int>>
  addAPoly(unsigned long int nPoly = 1,
           Game::EPECAddPolyMethod method = Game::EPECAddPolyMethod::sequential,
           std::set<std::vector<short int>> Polys = {});
  bool addThePoly(const unsigned long int &decimalEncoding);
  bool checkPolyFeas(const unsigned long int &decimalEncoding);
  bool checkPolyFeas(const std::vector<short int> &Fix);
  void clearPolyhedra() {
    this->Ai->clear();
    this->bi->clear();
    this->AllPolyhedra.clear();
  }
  polyLCP &addPolyFromX(const arma::vec &x, bool &ret);
  polyLCP &EnumerateAll(bool solveLP = true);
  std::string feas_detail_str() const;
  unsigned int getFeasiblePolyhedra() const { return this->feasiblePolyhedra; }
  void makeQP(Game::QP_objective &QP_obj, Game::QP_Param &QP) override;
};
} // namespace Game