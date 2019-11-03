#pragma once

/**
 * @file src/lcptolp.h To handle Linear Complementarity Problems.
 */

#include "epecsolve.h"
#include <armadillo>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>
#include <set>

// using namespace Game;

namespace Game {

arma::vec LPSolve(const arma::sp_mat &A, const arma::vec &b, const arma::vec &c,
                  int &status, bool Positivity = false);

unsigned int ConvexHull(const std::vector<arma::sp_mat *> *Ai,
                        const std::vector<arma::vec *> *bi, arma::sp_mat &A,
                        arma::vec &b, const arma::sp_mat Acom = {},
                        const arma::vec bcom = {});

void compConvSize(arma::sp_mat &A, const unsigned int nFinCons,
                  const unsigned int nFinVar,
                  const std::vector<arma::sp_mat *> *Ai,
                  const std::vector<arma::vec *> *bi, const arma::sp_mat &Acom,
                  const arma::vec &bcom);
/**
 * @brief Class to handle and solve linear complementarity problems
 */
/**
 * A class to handle linear complementarity problems (LCP)
 * especially as MIPs with bigM constraints
 * Also provides the convex hull of the feasible space, restricted feasible
 * space etc.
 */

class LCP {
  using spmat_Vec = std::vector<std::unique_ptr<arma::sp_mat>>;
  using vec_Vec = std::vector<std::unique_ptr<arma::vec>>;

private:
  // Essential data ironment for MIP/LP solves
  GRBEnv *env;    ///< Gurobi env
  arma::sp_mat M; ///< M in @f$Mx+q@f$ that defines the LCP
  arma::vec q;    ///< q in @f$Mx+q@f$ that defines the LCP
  perps Compl;    ///< Compl stores data in <Eqn, Var> form.
  unsigned int LeadStart{1}, LeadEnd{0}, nLeader{0};
  arma::sp_mat _A = {};
  arma::vec _b = {}; ///< Apart from @f$0 \le x \perp Mx+q\ge 0@f$, one needs@f$
                     ///< Ax\le b@f$ too!
  // Temporary data
  bool madeRlxdModel{false}; ///< Keep track if LCP::RlxdModel is made
  unsigned int nR, nC;
  int polyCounter{0};
  unsigned int feasiblePolyhedra{0};
  /// LCP feasible region is a union of polyhedra. Keeps track which of those
  /// inequalities are fixed to equality to get the individual polyhedra
  std::set<unsigned long int> AllPolyhedra =
      {}; ///< Decimal encoding of polyhedra that have been enumerated
  std::set<unsigned long int> feasiblePoly =
      {}; ///< Decimal encoding of polyhedra that have been enumerated
  std::set<unsigned long int> knownInfeas =
      {}; ///< Decimal encoding of polyhedra known to be infeasible
  std::set<unsigned long int> notProcessed =
      {}; ///< Decimal encoding of polyhedra to be processed
  unsigned long int maxTheoreticalPoly{0};
  std::unique_ptr<spmat_Vec>
      Ai; ///< Vector to contain the LHS of inner approx polyhedra
  std::unique_ptr<vec_Vec>
      bi;             ///< Vector to contain the RHS of inner approx polyhedra
  GRBModel RlxdModel; ///< A gurobi model with all complementarity constraints
                      ///< removed.

  bool errorCheck(bool throwErr = true) const;
  void defConst(GRBEnv *env);
  void makeRelaxed();
  void initializeNotProcessed() {
    const unsigned int nCompl = this->Compl.size();
    // 2^n - the number of polyhedra theoretically
    this->maxTheoreticalPoly = static_cast<unsigned long int>(pow(2, nCompl));
    for (unsigned long int i = 0; i < this->maxTheoreticalPoly; ++i)
      this->notProcessed.insert(i);
  }
  /* Solving relaxations and restrictions */
  std::unique_ptr<GRBModel> LCPasMIP(std::vector<unsigned int> FixEq = {},
                                     std::vector<unsigned int> FixVar = {},
                                     bool solve = false);
  std::unique_ptr<GRBModel> LCPasMIP(std::vector<short int> Fixes, bool solve);
  std::unique_ptr<GRBModel>
  LCP_Polyhed_fixed(std::vector<unsigned int> FixEq = {},
                    std::vector<unsigned int> FixVar = {});
  std::unique_ptr<GRBModel> LCP_Polyhed_fixed(arma::Col<int> FixEq,
                                              arma::Col<int> FixVar);
  template <class T> inline bool isZero(const T val) const {
    return (val >= -eps && val <= eps);
  }
  inline std::vector<short int> solEncode(GRBModel *model) const;
  std::vector<short int> solEncode(const arma::vec &x) const;
  std::vector<short int> solEncode(const arma::vec &z,
                                   const arma::vec &x) const;
  bool FixToPoly(const std::vector<short int> Fix, bool checkFeas = false,
                 bool custom = false, spmat_Vec *custAi = {},
                 vec_Vec *custbi = {});
  LCP &FixToPolies(const std::vector<short int> Fix, bool checkFeas = false,
                   bool custom = false, spmat_Vec *custAi = {},
                   vec_Vec *custbi = {});
  unsigned long int getNextPoly(Game::EPECAddPolyMethod method) const;

public:
  // Fudgible data
  long double bigM{1e7}; ///< bigM used to rewrite the LCP as MIP
  long double eps{
      1e-6}; ///< The threshold for optimality and feasability tollerances
  long double eps_int{1e-8}; ///< The threshold, below which a number would be
                             ///< considered to be zero.
  bool useIndicators{
      true}; ///< If true, complementarities will be handled with indicator
  ///< constraints. BigM formulation otherwise
  long int addPolyMethodSeed = {
      -1}; ///< Seeds the random generator for the random polyhedra selection.
           ///< Should be a positive value

  /** Constructors */
  /// Class has no default constructors
  LCP() = delete;

  LCP(GRBEnv *e)
      : env{e},
        RlxdModel(*e){}; ///< This constructor flor loading LCP from a file

  LCP(GRBEnv *env, arma::sp_mat M, arma::vec q, unsigned int LeadStart,
      unsigned LeadEnd, arma::sp_mat A = {},
      arma::vec b = {}); // Constructor with M,q,leader posn
  LCP(GRBEnv *env, arma::sp_mat M, arma::vec q, perps Compl,
      arma::sp_mat A = {},
      arma::vec b = {}); // Constructor with M, q, compl pairs
  LCP(GRBEnv *env, const Game::NashGame &N);
  LCP(const LCP &) = default;
  LCP &operator=(const LCP &) = default;

  /** Destructor - to delete the objects created with new operator */
  ~LCP();

  /** Return data and address */
  inline arma::sp_mat getM() { return this->M; } ///< Read-only access to LCP::M
  inline arma::sp_mat *getMstar() {
    return &(this->M);
  }                                           ///< Reference access to LCP::M
  inline arma::vec getq() { return this->q; } ///< Read-only access to LCP::q
  inline arma::vec *getqstar() {
    return &(this->q);
  } ///< Reference access to LCP::q
  inline unsigned int getLStart() {
    return LeadStart;
  } ///< Read-only access to LCP::LeadStart
  inline unsigned int getLEnd() {
    return LeadEnd;
  } ///< Read-only access to LCP::LeadEnd
  inline perps getCompl() {
    return this->Compl;
  }                                   ///< Read-only access to LCP::Compl
  void print(std::string end = "\n"); ///< Print a summary of the LCP
  inline unsigned int getNcol() { return this->M.n_cols; };

  inline unsigned int getNrow() { return this->M.n_rows; };

  bool extractSols(GRBModel *model, arma::vec &z, arma::vec &x,
                   bool extractZ = false) const;

  /* Getting single point solutions */
  std::unique_ptr<GRBModel> LCPasQP(bool solve = false);
  std::unique_ptr<GRBModel> LCPasMIP(bool solve = false);
  std::unique_ptr<GRBModel> MPECasMILP(const arma::sp_mat &C,
                                       const arma::vec &c,
                                       const arma::vec &x_minus_i,
                                       bool solve = false);
  std::unique_ptr<GRBModel>
  MPECasMIQP(const arma::sp_mat &Q, const arma::sp_mat &C, const arma::vec &c,
             const arma::vec &x_minus_i, bool solve = false);
  /* Convex hull computation */
  unsigned int ConvexHull(arma::sp_mat &A, arma::vec &b);
  unsigned int conv_Npoly() const;
  unsigned int conv_PolyPosition(const unsigned long int i) const;
  unsigned int conv_PolyWt(const unsigned long int i) const;

  std::set<unsigned long int> getAllPolyhedra() const {
    return this->AllPolyhedra;
  };
  std::set<unsigned long int> getnotProcessed() const {
    return this->notProcessed;
  };
  unsigned long int getNumTheoreticalPoly() const noexcept {
    return this->maxTheoreticalPoly;
  }

  LCP &makeQP(Game::QP_objective &QP_obj, Game::QP_Param &QP);

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
  LCP &addPolyFromX(const arma::vec &x, bool &ret);
  LCP &EnumerateAll(bool solveLP = true);
  std::string feas_detail_str() const;
  unsigned int getFeasiblePolyhedra() const { return this->feasiblePolyhedra; }
  void write(std::string filename, bool append = true) const;
  void save(std::string filename, bool erase = true) const;
  long int load(std::string filename, long int pos = 0);
};
} // namespace Game

/* Example for LCP  */
