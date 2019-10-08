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
  unsigned int LeadStart{1}, LeadEnd{0}, nLeader{0}, maxTheoreticalPoly{0};
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
  std::set<unsigned long int> knownInfeas =
      {}; ///< Decimal encoding of polyhedra known to be infeasible
  std::set<unsigned long int> notProcessed =
      {}; ///< Decimal encoding of polyhedra to be processed
  std::unique_ptr<spmat_Vec>
      Ai; ///< Vector to contain the LHS of inner approx polyhedra
  std::unique_ptr<vec_Vec>
      bi;             ///< Vector to contain the RHS of inner approx polyhedra
  GRBModel RlxdModel; ///< A gurobi model with all complementarity constraints
                      ///< removed.

  bool errorCheck(bool throwErr = true) const;

  void defConst(GRBEnv *env);

  void makeRelaxed();

  void initializeNotPorcessed() {
    const unsigned int nCompl = this->Compl.size();
    // 2^n - the number of polyhedra theoretically
    this->maxTheoreticalPoly = static_cast<unsigned int>(pow(2, nCompl));
    for (unsigned int i = 0; i < this->maxTheoreticalPoly; ++i)
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

  unsigned int getNextPoly(Game::EPECAddPolyMethod method) const;

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
  unsigned int
  ConvexHull(arma::sp_mat &A, ///< Convex hull inequality description LHS to
                              ///< be stored here
             arma::vec &b)    ///< Convex hull inequality description RHS to be
                              ///< stored here
  /**
   * Computes the convex hull of the feasible region of the LCP
   */
  {
    const std::vector<arma::sp_mat *> tempAi = [](spmat_Vec &uv) {
      std::vector<arma::sp_mat *> v{};
      for (const auto &x : uv)
        v.push_back(x.get());
      return v;
    }(*this->Ai);
    const std::vector<arma::vec *> tempbi = [](vec_Vec &uv) {
      std::vector<arma::vec *> v{};
      std::for_each(uv.begin(), uv.end(),
                    [&v](const std::unique_ptr<arma::vec> &ptr) {
                      v.push_back(ptr.get());
                    });
      return v;
    }(*this->bi);
    arma::sp_mat A_common;
    A_common = arma::join_cols(this->_A, -this->M);
    arma::vec b_common = arma::join_cols(this->_b, this->q);
    if (Ai->size() == 1) {
      A.zeros(Ai->at(0)->n_rows + A_common.n_rows,
              Ai->at(0)->n_cols + A_common.n_cols);
      b.zeros(bi->at(0)->n_rows + b_common.n_rows);
      A = arma::join_cols(*Ai->at(0), A_common);
      b = arma::join_cols(*bi->at(0), b_common);
      return 1;
    } else
      return Game::ConvexHull(&tempAi, &tempbi, A, b, A_common, b_common);
  };

  LCP &makeQP(Game::QP_objective &QP_obj, Game::QP_Param &QP);

  std::set<std::vector<short int>>
  addAPoly(unsigned int nPoly = 1,
           Game::EPECAddPolyMethod method = Game::EPECAddPolyMethod::sequential,
           std::set<std::vector<short int>> Polys = {});
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
/**
 * @page LCP_Example Game::LCP Example
 * Before reading this page, please ensure you are aware of the functionalities
 described in @link NashGame_Example Game::NashGame tutorial @endlink before
 following this page.
 *
 * Consider the Following linear complementarity problem with constraints
 * @f{eqnarray}{
 Ax + By \leq b\\
 0 \leq x \perp Mx + Ny + q \geq 0
 * @f}
 * These are the types of problems that are handled by the class Game::LCP but
 we use a different notation. Instead of using @p y to refer to the variables
 that don't have matching complementary equations, we call <i>all</i> the
 variables as @p x and we keep track of the position of variables which are not
 complementary to any equation.
 *
 * <b>Points to note: </b>
 * - The set of indices of @p x which are not complementary to any equation
 should be a consecutive set of indices. For consiceness, these components will
 be called as <i>Leader vars components</i> of @p x.
 * - Suppose the leader vars components of @p x are removed from @p x, in the
 remaining components, the first component should be complementary to the first
 row defined by @p M, second component should be complementary to the second row
 defined by @p M and so on.
 *
 * Now consider the following linear complementarity problem.
 * @f{align*}{
        x_1 + x_2 + x_3 \le 12\\
        0\le x_1 \perp x_4 - 1 \ge 0\\
        0\le x_2 \le 2 \\
        0 \le x_3 \perp 2x_3 + x_5 \ge 0\\
        0 \le x_4 \perp -x_1 + x_2 + 10 \ge 0\\
        0 \le x_5 \perp x_2 - x_3 + 5 \ge 0
 * @f}
 * Here indeed @f$ x_2 @f$ is the leader vars component with no complementarity
 equation. This problem can be entered into the Game::LCP class as follows.
 * @code
                arma::sp_mat M(4, 5); // We have four complementarity eqns and 5
 variables. arma::vec q(4); M.zeros();
                // First eqn
                M(0, 3) = 1;
                q(0) = -1;
                // Second eqn
                M(1, 2) = 2;
                M(1, 4)  = 1;
                q(1) = 0;
                // Third eqn
                M(2, 0) = -1;
                M(2, 1) = 1;
                q(2) = 10;
                // Fourth eqn
                M(3, 1) = 1 ;
                M(3, 2) = -1;
                q(3) = 5;
                // Other common constraints
                arma::sp_mat A(2, 5); arma::vec b;
                A.zeros();
                // x_2 <= 2 constraint
                A(0, 1) = 1;
                b(0) = 2;
                // x_1 + x_2 + x_3 <= 12 constraint
                A(1, 0) = 1;
                A(1, 1) = 1;
                A(1, 2) = 1;
                b(1) = 12;
 * @endcode
 *
 * Now, since the variable with no complementarity pair is @f$x_2@f$ which is in
 position @p 1 (counting from 0) of the vector @p x, the arguments @p LeadStart
 and @p LeadEnd in the constructor, Game::LCP::LCP are @p 1 as below.
 * @code
                GRBEnv env;
                LCP lcp = LCP(&env, M, q, 1, 1, A, b);
 * @endcode
 * This problem can be solved either using big-M based disjunctive formulation
 with the value of the @p bigM can also be chosen. But a more preferred means of
 solving is by using indicator constraints, where the algorithm tries to
 automatically identify good choices of bigM for each disjunction. Use the
 former option, only if you are very confident of  your choice of a small value
 of @p bigM.
 * @code
 // Solve using bigM constraints
 lcp.useIndicators = false;
 lcp.bigM = 1e5;
 auto bigMModel = lcp.LCPasMIP(true);

 // Solve using indicator constraints
 lcp.useIndicators = true;
 auto indModel = lcp.LCPasMIP(true);
 * @endcode
 * Both @p bigMModel and @p indModel are std::unique_ptr  to GRBModel objects.
 So all native gurobi operations can be performed on these objects.
 *
 * This LCP as multiple solutions. In fact the solution set can be parameterized
 as below.
 * @f{align}{
 x_1 &= 10 + t\\
 x_2 &= t\\
 x_3 &= 0\\
 x_4 &= 1\\
 x_5 &= 0
 @f}
 * for @f$t \in [0, 1]@f$.
 *
 But some times, one might want to solve an MPEC. i.e., optimize over the
 feasible region of the set as decribed above. For this purpose, two functions
 Game::LCP::MPECasMILP and Game::LCP::MPECasMIQP are available, depending upon
 whether one wants to optimize a linear objective function or a convex quadratic
 objective function over the set of solutions.
 *
 *
 *
 */
