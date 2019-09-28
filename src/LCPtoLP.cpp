#include "lcptolp.h"
#include <algorithm>
#include <armadillo>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <cmath>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>

using namespace std;
using namespace Utils;

bool operator==(vector<short int> Fix1, vector<short int> Fix2)
/**
 * @brief Checks if two vector<int> are of same size and hold same values in the
 * same order
 * @warning Might be deprecated, as it pollutes global namespaces
 * @returns @p true if Fix1 and Fix2 have the same elements else @p false
 */
{
  if (Fix1.size() != Fix2.size())
    return false;
  for (unsigned int i = 0; i < Fix1.size(); i++) {
    if (Fix1.at(i) != Fix2.at(i))
      return false;
  }
  return true;
}

bool operator<(vector<short int> Fix1, vector<short int> Fix2)
/**
 * @details \b GrandParent:
 *  	Either the same value as the grand child, or has 0 in that location
 *
 *  \b Grandchild:
 *  	Same val as grand parent in every location, except any val allowed, if
 * grandparent is 0
 * @warning Might be deprecated, as it pollutes global namespaces
 * @returns @p true if Fix1 is (grand) child of Fix2
 */
{
  if (Fix1.size() != Fix2.size())
    return false;
  for (unsigned int i = 0; i < Fix1.size(); i++) {
    if (Fix1.at(i) != Fix2.at(i) && Fix1.at(i) * Fix2.at(i) != 0) {
      return false; // Fix1 is not a child of Fix2
    }
  }
  return true; // Fix1 is a child of Fix2
}

bool operator>(vector<int> Fix1, vector<int> Fix2) { return (Fix2 < Fix1); }

void Game::LCP::defConst(GRBEnv *env)
/**
 * @brief Assign default values to LCP attributes
 * @details Internal member that can be called from multiple constructors
 * to assign default values to some attributes of the class.
 */
{
  this->Ai = unique_ptr<spmat_Vec>(new spmat_Vec());
  this->bi = unique_ptr<vec_Vec>(new vec_Vec());
  this->RlxdModel.set(GRB_IntParam_OutputFlag, VERBOSE);
  this->env = env;
  this->nR = this->M.n_rows;
  this->nC = this->M.n_cols;
}

Game::LCP::LCP(
    GRBEnv *env,    ///< Gurobi environment required
    arma::sp_mat M, ///< @p M in @f$Mx+q@f$
    arma::vec q,    ///< @p q in @f$Mx+q@f$
    perps Compl,    ///< Pairing equations and variables for complementarity
    arma::sp_mat A, ///< Any equations without a complemntarity variable
    arma::vec b     ///< RHS of equations without complementarity variables
    )
    : M{M}, q{q}, _A{A}, _b{b}, RlxdModel(*env)
/// @brief Constructor with M, q, compl pairs
{
  defConst(env);
  this->Compl = perps(Compl);
  std::sort(
      this->Compl.begin(), this->Compl.end(),
      [](pair<unsigned int, unsigned int> a,
         pair<unsigned int, unsigned int> b) { return a.first < b.first; });
  for (auto p : this->Compl)
    if (p.first != p.second) {
      this->LeadStart = p.first;
      this->LeadEnd = p.second - 1;
      this->nLeader = this->LeadEnd - this->LeadStart + 1;
      this->nLeader = this->nLeader > 0 ? this->nLeader : 0;
      break;
    }
  this->initializeNotPorcessed();
}

Game::LCP::LCP(
    GRBEnv *env,            ///< Gurobi environment required
    arma::sp_mat M,         ///< @p M in @f$Mx+q@f$
    arma::vec q,            ///< @p q in @f$Mx+q@f$
    unsigned int LeadStart, ///< Position where variables which are not
                            ///< complementary to any equation starts
    unsigned LeadEnd, ///< Position where variables which are not complementary
                      ///< to any equation ends
    arma::sp_mat A,   ///< Any equations without a complemntarity variable
    arma::vec b       ///< RHS of equations without complementarity variables
    )
    : M{M}, q{q}, _A{A}, _b{b}, RlxdModel(*env)
/// @brief Constructor with M,q,leader posn
/**
 * @warning This might be deprecated to support LCP functioning without sticking
 * to the output format of NashGame
 */
{
  defConst(env);
  this->LeadStart = LeadStart;
  this->LeadEnd = LeadEnd;
  this->nLeader = this->LeadEnd - this->LeadStart + 1;
  this->nLeader = this->nLeader > 0 ? this->nLeader : 0;
  for (unsigned int i = 0; i < M.n_rows; i++) {
    unsigned int count = i < LeadStart ? i : i + nLeader;
    this->Compl.push_back({i, count});
  }
  std::sort(
      this->Compl.begin(), this->Compl.end(),
      [](pair<unsigned int, unsigned int> a,
         pair<unsigned int, unsigned int> b) { return a.first < b.first; });
  this->initializeNotPorcessed();
}

Game::LCP::LCP(GRBEnv *env, const NashGame &N)
    : RlxdModel(*env)
/**
 *	@brief Constructer given a NashGame
 *	@details Given a NashGame, computes the KKT of the lower levels, and
 *makes the appropriate LCP object.
 *
 *	This constructor is the most suited for highlevel usage.
 *	@note Most preferred constructor for user interface.
 */
{
  arma::sp_mat M;
  arma::vec q;
  perps Compl;
  N.FormulateLCP(M, q, Compl);
  // LCP(env, M, q, Compl, N.RewriteLeadCons(), N.getMCLeadRHS());

  this->M = M;
  this->q = q;
  this->_A = N.RewriteLeadCons();
  this->_b = N.getMCLeadRHS();
  defConst(env);
  this->Compl = perps(Compl);
  sort(this->Compl.begin(), this->Compl.end(),
       [](pair<unsigned int, unsigned int> a,
          pair<unsigned int, unsigned int> b) { return a.first < b.first; });
  // Delete no more!
  for (auto p : this->Compl) {
    if (p.first != p.second) {
      this->LeadStart = p.first;
      this->LeadEnd = p.second - 1;
      this->nLeader = this->LeadEnd - this->LeadStart + 1;
      this->nLeader = this->nLeader > 0 ? this->nLeader : 0;
      break;
    }
  }
  this->initializeNotPorcessed();
}

Game::LCP::~LCP()
/** @brief Destructor of LCP */
/** LCP object owns the pointers to definitions of its polyhedra that it owns
 It has to be deleted and freed. */
{}

void Game::LCP::makeRelaxed()
/** @brief Makes a Gurobi object that relaxes complementarity constraints in an
   LCP */
/** @details A Gurobi object is stored in the LCP object, that has all
 * complementarity constraints removed. A copy of this object is used by other
 * member functions */
{
  try {
    if (this->madeRlxdModel)
      return;
    BOOST_LOG_TRIVIAL(trace)
        << "Game::LCP::makeRelaxed: Creating a model with : " << nR
        << " variables and  " << nC << " constraints";
    GRBVar x[nC], z[nR];
    BOOST_LOG_TRIVIAL(trace)
        << "Game::LCP::makeRelaxed: Initializing variables";
    for (unsigned int i = 0; i < nC; i++)
      x[i] = RlxdModel.addVar(0, GRB_INFINITY, 1, GRB_CONTINUOUS,
                              "x_" + to_string(i));
    for (unsigned int i = 0; i < nR; i++)
      z[i] = RlxdModel.addVar(0, GRB_INFINITY, 1, GRB_CONTINUOUS,
                              "z_" + to_string(i));
    BOOST_LOG_TRIVIAL(trace) << "Game::LCP::makeRelaxed: Added variables";
    for (unsigned int i = 0; i < nR; i++) {
      GRBLinExpr expr = 0;
      for (auto v = M.begin_row(i); v != M.end_row(i); ++v)
        expr += (*v) * x[v.col()];
      expr += q(i);
      RlxdModel.addConstr(expr, GRB_EQUAL, z[i], "z_" + to_string(i) + "_def");
    }
    BOOST_LOG_TRIVIAL(trace)
        << "Game::LCP::makeRelaxed: Added equation definitions";
    // If @f$Ax \leq b@f$ constraints are there, they should be included too!
    if (this->_A.n_nonzero != 0 && this->_b.n_rows != 0) {
      if (_A.n_cols != nC || _A.n_rows != _b.n_rows) {
        BOOST_LOG_TRIVIAL(trace) << "(" << _A.n_rows << "," << _A.n_cols
                                 << ")\t" << _b.n_rows << " " << nC;
        throw string("A and b are incompatible! Thrown from makeRelaxed()");
      }
      for (unsigned int i = 0; i < _A.n_rows; i++) {
        GRBLinExpr expr = 0;
        for (auto a = _A.begin_row(i); a != _A.end_row(i); ++a)
          expr += (*a) * x[a.col()];
        RlxdModel.addConstr(expr, GRB_LESS_EQUAL, _b(i),
                            "commonCons_" + to_string(i));
      }
      BOOST_LOG_TRIVIAL(trace)
          << "Game::LCP::makeRelaxed: Added common constraints";
    }
    RlxdModel.update();
    this->madeRlxdModel = true;
  } catch (const char *e) {
    cerr << "Error in Game::LCP::makeRelaxed: " << e << '\n';
    throw;
  } catch (string e) {
    cerr << "String: Error in Game::LCP::makeRelaxed: " << e << '\n';
    throw;
  } catch (exception &e) {
    cerr << "Exception: Error in Game::LCP::makeRelaxed: " << e.what() << '\n';
    throw;
  } catch (GRBException &e) {
    cerr << "GRBException: Error in Game::LCP::makeRelaxed: "
         << e.getErrorCode() << "; " << e.getMessage() << '\n';
    throw;
  }
}

unique_ptr<GRBModel> Game::LCP::LCP_Polyhed_fixed(
    vector<unsigned int>
        FixEq, ///< If index is present, equality imposed on that variable
    vector<unsigned int>
        FixVar ///< If index is present, equality imposed on that equation
    )
/**
 * The returned model has constraints
 * corresponding to the indices in FixEq set to equality
 * and variables corresponding to the indices
 * present in FixVar set to equality (=0)
 * @note This model returned could either be a relaxation or a restriction or
 * neither. If every index is present in at least one of the two vectors --- @p
 * FixEq or @p FixVar --- then it is a restriction.
 * @note <tt>LCP::LCP_Polyhed_fixed({},{})</tt> is equivalent to accessing
 * LCP::RlxdModel
 * @warning The FixEq and FixVar variables are used under a different convention
 * here!
 * @warning Note that the model returned by this function has to be explicitly
 * deleted using the delete operator.
 * @returns unique pointer to a GRBModel
 */
{
  makeRelaxed();
  unique_ptr<GRBModel> model(new GRBModel(this->RlxdModel));
  for (auto i : FixEq) {
    if (i >= nR)
      throw "Element in FixEq is greater than nC";
    model->getVarByName("z_" + to_string(i)).set(GRB_DoubleAttr_UB, 0);
  }
  for (auto i : FixVar) {
    if (i >= nC)
      throw "Element in FixEq is greater than nC";
    model->getVarByName("z_" + to_string(i)).set(GRB_DoubleAttr_UB, 0);
  }
  return model;
}

unique_ptr<GRBModel> Game::LCP::LCP_Polyhed_fixed(
    arma::Col<int> FixEq, ///< If non zero, equality imposed on variable
    arma::Col<int> FixVar ///< If non zero, equality imposed on equation
    )
/**
 * Returs a model created from a given model
 * The returned model has constraints
 * corresponding to the non-zero elements of FixEq set to equality
 * and variables corresponding to the non-zero
 * elements of FixVar set to equality (=0)
 * @note This model returned could either be a relaxation or a restriction or
 * neither.  If FixEq + FixVar is at least 1 (element-wise), then it is a
 * restriction.
 * @note <tt>LCP::LCP_Polyhed_fixed({0,...,0},{0,...,0})</tt> is equivalent to
 * accessing LCP::RlxdModel
 * @warning Note that the model returned by this function has to be explicitly
 * deleted using the delete operator.
 * @returns unique pointer to a GRBModel
 */
{
  makeRelaxed();
  unique_ptr<GRBModel> model{new GRBModel(this->RlxdModel)};
  for (unsigned int i = 0; i < nC; i++)
    if (FixVar[i])
      model->getVarByName("x_" + to_string(i)).set(GRB_DoubleAttr_UB, 0);
  for (unsigned int i = 0; i < nR; i++)
    if (FixEq[i])
      model->getVarByName("z_" + to_string(i)).set(GRB_DoubleAttr_UB, 0);
  model->update();
  return model;
}

unique_ptr<GRBModel> Game::LCP::LCPasMIP(
    vector<short int>
        Fixes, ///< For each Variable, +1 fixes the equation to equality and -1
               ///< fixes the variable to equality. A value of 0 fixes neither.
    bool solve ///< Whether the model is to be solved before returned
    )
/**
 * Uses the big M method to solve the complementarity problem. The variables and
 * eqns to be set to equality can be given in Fixes in 0/+1/-1 notation
 * @note Returned model is \e always a restriction. For <tt>Fixes =
 * {0,...,0}</tt>, the returned model would solve the exact LCP.
 * @throws string if <tt> Fixes.size()!= </tt> number of equations (for
 * complementarity).
 * @warning Note that the model returned by this function has to be explicitly
 * deleted using the delete operator.
 * @returns unique pointer to a GRBModel
 */
{
  if (Fixes.size() != this->nR)
    throw string("Bad size for Fixes in Game::LCP::LCPasMIP");
  vector<unsigned int> FixVar, FixEq;
  for (unsigned int i = 0; i < nR; i++) {
    if (Fixes[i] == 1)
      FixEq.push_back(i);
    if (Fixes[i] == -1)
      FixVar.push_back(i > this->LeadStart ? i + this->nLeader : i);
  }
  return this->LCPasMIP(FixEq, FixVar, solve);
}

unique_ptr<GRBModel> Game::LCP::LCPasMIP(
    vector<unsigned int> FixEq,  ///< If any equation is to be fixed to equality
    vector<unsigned int> FixVar, ///< If any variable is to be fixed to equality
    bool solve ///< Whether the model should be solved in the function before
               ///< returned.
    )
/**
 * Uses the big M method to solve the complementarity problem. The variables and
 * eqns to be set to equality can be given in FixVar and FixEq.
 * @note Returned model is \e always a restriction. For <tt>FixEq = FixVar =
 * {}</tt>, the returned model would solve the exact LCP.
 * @warning Note that the model returned by this function has to be explicitly
 * deleted using the delete operator.
 * @returns unique pointer to a GRBModel
 */
{
  makeRelaxed();
  unique_ptr<GRBModel> model{new GRBModel(this->RlxdModel)};
  // Creating the model
  try {
    GRBVar x[nC], z[nR], u[nR], v[nR];
    // Get hold of the Variables and Eqn Variables
    for (unsigned int i = 0; i < nC; i++)
      x[i] = model->getVarByName("x_" + to_string(i));
    for (unsigned int i = 0; i < nR; i++)
      z[i] = model->getVarByName("z_" + to_string(i));
    // Define binary variables for bigM
    for (unsigned int i = 0; i < nR; i++)
      u[i] = model->addVar(0, 1, 0, GRB_BINARY, "u_" + to_string(i));
    if (this->useIndicators)
      for (unsigned int i = 0; i < nR; i++)
        v[i] = model->addVar(0, 1, 0, GRB_BINARY, "v_" + to_string(i));
    // Include ALL Complementarity constraints using bigM

    if (this->useIndicators) {
      BOOST_LOG_TRIVIAL(trace)
          << "Using indicator constraints for complementarities.";
    } else {
      BOOST_LOG_TRIVIAL(trace)
          << "Using bigM for complementarities with M=" << this->bigM;
    }

    GRBLinExpr expr = 0;
    for (const auto p : Compl) {
      // z[i] <= Mu constraint

      // u[j]=0 --> z[i] <=0
      if (!this->useIndicators) {
        expr = bigM * u[p.first];
        model->addConstr(expr, GRB_GREATER_EQUAL, z[p.first],
                         "z" + to_string(p.first) + "_L_Mu" +
                             to_string(p.first));
      } else {
        model->addGenConstrIndicator(
            u[p.first], 1, z[p.first], GRB_LESS_EQUAL, 0,
            "z_ind_" + to_string(p.first) + "_L_Mu_" + to_string(p.first));
      }
      // x[i] <= M(1-u) constraint
      if (!this->useIndicators) {
        expr = bigM - bigM * u[p.first];
        model->addConstr(expr, GRB_GREATER_EQUAL, x[p.second],
                         "x" + to_string(p.first) + "_L_MuDash" +
                             to_string(p.first));
      } else {
        model->addGenConstrIndicator(
            v[p.first], 1, x[p.second], GRB_LESS_EQUAL, 0,
            "x_ind_" + to_string(p.first) + "_L_MuDash_" + to_string(p.first));
      }

      if (this->useIndicators)
        model->addConstr(u[p.first] + v[p.first], GRB_EQUAL, 1,
                         "uv_sum_" + to_string(p.first));
    }
    // If any equation or variable is to be fixed to zero, that happens here!
    for (auto i : FixVar)
      model->addConstr(x[i], GRB_EQUAL, 0.0);
    for (auto i : FixEq)
      model->addConstr(z[i], GRB_EQUAL, 0.0);
    model->update();
    if (!this->useIndicators) {
      model->set(GRB_DoubleParam_IntFeasTol, this->eps_int);
      model->set(GRB_DoubleParam_FeasibilityTol, this->eps);
      model->set(GRB_DoubleParam_OptimalityTol, this->eps);
    }
    // Get first Equilibrium
    model->set(GRB_IntParam_SolutionLimit, 1);
    if (solve)
      model->optimize();
    return model;
  } catch (const char *e) {
    cerr << "Error in Game::LCP::LCPasMIP: " << e << '\n';
    throw;
  } catch (string e) {
    cerr << "String: Error in Game::LCP::LCPasMIP: " << e << '\n';
    throw;
  } catch (exception &e) {
    cerr << "Exception: Error in Game::LCP::LCPasMIP: " << e.what() << '\n';
    throw;
  } catch (GRBException &e) {
    cerr << "GRBException: Error in Game::LCP::LCPasMIP: " << e.getErrorCode()
         << "; " << e.getMessage() << '\n';
    throw;
  }
  return nullptr;
}

bool Game::LCP::errorCheck(
    bool throwErr ///< If this is true, function throws an
                  ///< error, else, it just returns false
    ) const
/**
 * Checks if the `M` and `q` given to create the LCP object are of
 * compatible size, given the number of leader variables
 */
{
  const unsigned int nR = M.n_rows;
  const unsigned int nC = M.n_cols;
  if (throwErr) {
    if (nR != q.n_rows)
      throw "M and q have unequal number of rows";
    if (nR + nLeader != nC)
      throw "Inconsistency between number of leader vars " +
          to_string(nLeader) + ", number of rows " + to_string(nR) +
          " and number of cols " + to_string(nC);
  }
  return (nR == q.n_rows && nR + nLeader == nC);
}

void Game::LCP::print(string end) {
  cout << "LCP with " << this->nR << " rows and " << this->nC << " columns."
       << end;
}

unsigned int Game::ConvexHull(
    const vector<arma::sp_mat *>
        *Ai, ///< Inequality constraints LHS that define polyhedra whose convex
             ///< hull is to be found
    const vector<arma::vec *>
        *bi,         ///< Inequality constraints RHS that define
                     ///< polyhedra whose convex hull is to be found
    arma::sp_mat &A, ///< Pointer to store the output of the convex hull LHS
    arma::vec &b,    ///< Pointer to store the output of the convex hull RHS
    const arma::sp_mat
        Acom,            ///< any common constraints to all the polyhedra - lhs.
    const arma::vec bcom ///< Any common constraints to ALL the polyhedra - RHS.
    )
/** @brief Computing convex hull of finite unioon of polyhedra
 * @details Computes the convex hull of a finite union of polyhedra where
 * each polyhedra @f$P_i@f$ is of the form
 * @f{eqnarray}{
 * A^ix &\leq& b^i\\
 * x &\geq& 0
 * @f}
 * This uses Balas' approach to compute the convex hull.
 *
 * <b>Cross reference:</b> Conforti, Michele; Cornuéjols, Gérard; and Zambelli,
 * Giacomo. Integer programming. Vol. 271. Berlin: Springer, 2014. Refer:
 * Eqn 4.31
 */
{
  // Count number of polyhedra and the space we are in!
  const unsigned int nPoly{static_cast<unsigned int>(Ai->size())};
  // Error check
  if (nPoly == 0)
    throw string("Empty vector of polyhedra given! Problem might be "
                 "infeasible."); // There should be at least 1 polyhedron to
                                 // consider
  const unsigned int nC{static_cast<unsigned int>(Ai->front()->n_cols)};
  const unsigned int nComm{static_cast<unsigned int>(Acom.n_rows)};

  if (nComm > 0 && Acom.n_cols != nC)
    throw string("Inconsistent number of variables in the common polyhedron");
  if (nComm > 0 && nComm != bcom.n_rows)
    throw string(
        "Inconsistent number of rows in LHS and RHS in the common polyhedron");

  // Count the number of variables in the convex hull.
  unsigned int nFinCons{0}, nFinVar{0};
  if (nPoly != bi->size())
    throw string("Inconsistent number of LHS and RHS for polyhedra");
  for (unsigned int i = 0; i != nPoly; i++) {
    if (Ai->at(i)->n_cols != nC)
      throw string("Inconsistent number of variables in the polyhedra ") +
          to_string(i) + "; " + to_string(Ai->at(i)->n_cols) +
          "!=" + to_string(nC);
    if (Ai->at(i)->n_rows != bi->at(i)->n_rows)
      throw string("Inconsistent number of rows in LHS and RHS of polyhedra ") +
          to_string(i) + ";" + to_string(Ai->at(i)->n_rows) +
          "!=" + to_string(bi->at(i)->n_rows);
    nFinCons += Ai->at(i)->n_rows;
  }
  // For common constraint copy
  nFinCons += nPoly * nComm;

  const unsigned int FirstCons = nFinCons;

  // 2nd constraint in Eqn 4.31 of Conforti - twice so we have 2 ineq instead of
  // 1 eq constr
  nFinCons += nC * 2;
  // 3rd constr in Eqn 4.31. Again as two ineq constr.
  nFinCons += 2;
  // Common constraints
  // nFinCons += Acom.n_rows;

  nFinVar = nPoly * nC + nPoly +
            nC; // All x^i variables + delta variables+ original x variables
  A.zeros(nFinCons, nFinVar);
  b.zeros(nFinCons);
  // A.zeros(nFinCons, nFinVar); b.zeros(nFinCons);
  // Implements the first constraint more efficiently using better constructors
  // for sparse matrix
  Game::compConvSize(A, nFinCons, nFinVar, Ai, bi, Acom, bcom);

  // Counting rows completed
  /****************** SLOW LOOP BEWARE *******************/
  for (unsigned int i = 0; i < nPoly; i++) {
    BOOST_LOG_TRIVIAL(trace) << "Game::ConvexHull: Handling Polyhedron "
                             << i + 1 << " out of " << nPoly;
    // First constraint in (4.31)
    // A.submat(complRow, i*nC, complRow+nConsInPoly-1, (i+1)*nC-1) =
    // *Ai->at(i); // Slowest line. Will arma improve this? First constraint RHS
    // A.submat(complRow, nPoly*nC+i, complRow+nConsInPoly-1, nPoly*nC+i) =
    // -*bi->at(i); Second constraint in (4.31)
    for (unsigned int j = 0; j < nC; j++) {
      A.at(FirstCons + 2 * j, nC + (i * nC) + j) = 1;
      A.at(FirstCons + 2 * j + 1, nC + (i * nC) + j) = -1;
    }
    // Third constraint in (4.31)
    A.at(FirstCons + nC * 2, nC + nPoly * nC + i) = 1;
    A.at(FirstCons + nC * 2 + 1, nC + nPoly * nC + i) = -1;
  }
  /****************** SLOW LOOP BEWARE *******************/
  // Second Constraint RHS
  for (unsigned int j = 0; j < nC; j++) {
    A.at(FirstCons + 2 * j, j) = -1;
    A.at(FirstCons + 2 * j + 1, j) = 1;
  }
  // Third Constraint RHS
  b.at(FirstCons + nC * 2) = 1;
  b.at(FirstCons + nC * 2 + 1) = -1;
  return nPoly; ///< Perfrorm increasingly better inner approximations in
                ///< iterations
}

void Game::compConvSize(
    arma::sp_mat &A,             ///< Output parameter
    const unsigned int nFinCons, ///< Number of rows in final matrix A
    const unsigned int nFinVar,  ///< Number of columns in the final matrix A
    const vector<arma::sp_mat *>
        *Ai, ///< Inequality constraints LHS that define polyhedra whose convex
    ///< hull is to be found
    const vector<arma::vec *>
        *bi, ///< Inequality constraints RHS that define
             ///< polyhedra whose convex hull is to be found
    const arma::sp_mat
        &Acom,            ///< LHS of the common constraints for all polyhedra
    const arma::vec &bcom ///< RHS of the common constraints for all polyhedra
    )
/**
 * @brief INTERNAL FUNCTION NOT FOR GENERAL USE.
 * @warning INTERNAL FUNCTION NOT FOR GENERAL USE.
 * @internal To generate the matrix "A" in Game::ConvexHull using batch
 * insertion constructors. This is faster than the original line in the code:
 * A.submat(complRow, i*nC, complRow+nConsInPoly-1, (i+1)*nC-1) = *Ai->at(i);
 * Motivation behind this: Response from
 * armadillo:-https://gitlab.com/conradsnicta/armadillo-code/issues/111
 */
{
  const unsigned int nPoly{static_cast<unsigned int>(Ai->size())};
  const unsigned int nC{static_cast<unsigned int>(Ai->front()->n_cols)};
  unsigned int N{0}; // Total number of nonzero elements in the final matrix
  const unsigned int nCommnz{
      static_cast<unsigned int>(Acom.n_nonzero + bcom.n_rows)};
  for (unsigned int i = 0; i < nPoly; i++) {
    N += Ai->at(i)->n_nonzero;
    N += bi->at(i)->n_rows;
  }
  N += nCommnz *
       nPoly; // The common constraints have to be copied for each polyhedron.

  // Now computed N which is the total number of nonzeros.
  arma::umat locations; // location of nonzeros
  arma::vec val;        // nonzero values
  locations.zeros(2, N);
  val.zeros(N);

  unsigned int count{0}, rowCount{0}, colCount{nC};
  for (unsigned int i = 0; i < nPoly; i++) {
    for (auto it = Ai->at(i)->begin(); it != Ai->at(i)->end();
         ++it) // First constraint
    {
      locations(0, count) = rowCount + it.row();
      locations(1, count) = colCount + it.col();
      val(count) = *it;
      ++count;
    }
    for (unsigned int j = 0; j < bi->at(i)->n_rows;
         ++j) // RHS of first constraint
    {
      locations(0, count) = rowCount + j;
      locations(1, count) = nC + nC * nPoly + i;
      val(count) = -bi->at(i)->at(j);
      ++count;
    }
    rowCount += Ai->at(i)->n_rows;

    // For common constraints
    for (auto it = Acom.begin(); it != Acom.end(); ++it) // First constraint
    {
      locations(0, count) = rowCount + it.row();
      locations(1, count) = colCount + it.col();
      val(count) = *it;
      ++count;
    }
    for (unsigned int j = 0; j < bcom.n_rows; ++j) // RHS of first constraint
    {
      locations(0, count) = rowCount + j;
      locations(1, count) = nC + nC * nPoly + i;
      val(count) = -bcom.at(j);
      ++count;
    }
    rowCount += Acom.n_rows;

    colCount += nC;
  }
  A = arma::sp_mat(locations, val, nFinCons, nFinVar);
}

arma::vec
Game::LPSolve(const arma::sp_mat &A, ///< The constraint matrix
              const arma::vec &b,    ///< RHS of the constraint matrix
              const arma::vec &c,    ///< If feasible, returns a vector that
                                     ///< minimizes along this direction
              int &status, ///< Status of the optimization problem. If optimal,
                           ///< this will be GRB_OPTIMAL
              bool Positivity ///< Should @f$x\geq0@f$ be enforced?
              )
/**
 Checks if the polyhedron given by @f$ Ax\leq b@f$ is feasible.
 If yes, returns the point @f$x@f$ in the polyhedron that minimizes @f$c^Tx@f$
 Positivity can be enforced on the variables easily.
*/
{
  unsigned int nR, nC;
  nR = A.n_rows;
  nC = A.n_cols;
  if (c.n_rows != nC)
    throw "Inconsistency in no of Vars in isFeas()";
  if (b.n_rows != nR)
    throw "Inconsistency in no of Constr in isFeas()";

  arma::vec sol = arma::vec(c.n_rows, arma::fill::zeros);
  const double lb = Positivity ? 0 : -GRB_INFINITY;

  GRBEnv env;
  GRBModel model = GRBModel(env);
  GRBVar x[nC];
  GRBConstr a[nR];
  // Adding Variables
  for (unsigned int i = 0; i < nC; i++)
    x[i] = model.addVar(lb, GRB_INFINITY, c.at(i), GRB_CONTINUOUS,
                        "x_" + to_string(i));
  // Adding constraints
  for (unsigned int i = 0; i < nR; i++) {
    GRBLinExpr lin{0};
    for (auto j = A.begin_row(i); j != A.end_row(i); ++j)
      lin += (*j) * x[j.col()];
    a[i] = model.addConstr(lin, GRB_LESS_EQUAL, b.at(i));
  }
  model.set(GRB_IntParam_OutputFlag, VERBOSE);
  model.set(GRB_IntParam_DualReductions, 0);
  model.optimize();
  status = model.get(GRB_IntAttr_Status);
  if (status == GRB_OPTIMAL)
    for (unsigned int i = 0; i < nC; i++)
      sol.at(i) = x[i].get(GRB_DoubleAttr_X);
  return sol;
}

bool Game::LCP::extractSols(
    GRBModel *model, ///< The Gurobi Model that was solved (perhaps using
    ///< Game::LCP::LCPasMIP)
    arma::vec &z, ///< Output variable - where the equation values are stored
    arma::vec &x, ///< Output variable - where the variable values are stored
    bool extractZ ///< z values are filled only if this is true
    ) const
/** @brief Extracts variable and equation values from a solved Gurobi model for
   LCP */
/** @warning This solves the model if the model is not already solve */
/** @returns @p false if the model is not solved to optimality. @p true
   otherwise */
{
  if (model->get(GRB_IntAttr_Status) == GRB_LOADED)
    model->optimize();
  auto status = model->get(GRB_IntAttr_Status);
  if (!(status == GRB_OPTIMAL || status == GRB_SUBOPTIMAL ||
        status == GRB_SOLUTION_LIMIT))
    return false;
  x.zeros(nC);
  if (extractZ)
    z.zeros(nR);
  for (unsigned int i = 0; i < nR; i++) {
    x[i] = model->getVarByName("x_" + to_string(i)).get(GRB_DoubleAttr_X);
    if (extractZ)
      z[i] = model->getVarByName("z_" + to_string(i)).get(GRB_DoubleAttr_X);
  }
  for (unsigned int i = nR; i < nC; i++)
    x[i] = model->getVarByName("x_" + to_string(i)).get(GRB_DoubleAttr_X);
  return true;
}

std::vector<short int> Game::LCP::solEncode(const arma::vec &x) const
/// @brief Given variable values, encodes it in 0/+1/-1
/// format and returns it.
/// @details Gives the 0/+1/-1 notation. The notation is defined as follows.
/// Note that, if the input is feasible, then in each complementarity pair (Eqn,
/// Var), at least one of the two is zero.
///
/// - If the equation is zero in a certain index and the variable is non-zero,
/// then that index is noted by +1.
/// - If the variable is zero in a certain index and the equation is non-zero,
/// then that index is noted by +1.
/// - If both the variable and equation are zero, then that index is noted by 0.
{
  return this->solEncode(this->M * x + this->q, x);
}

vector<short int> Game::LCP::solEncode(const arma::vec &z, ///< Equation values
                                       const arma::vec &x  ///< Variable values
                                       ) const
/// @brief Given variable values and equation values, encodes it in 0/+1/-1
/// format and returns it.
{
  vector<short int> solEncoded(nR, 0);
  for (const auto p : Compl) {
    unsigned int i, j;
    i = p.first;
    j = p.second;
    if (isZero(z(i)))
      solEncoded.at(i)++;
    if (isZero(x(j)))
      solEncoded.at(i)--;
    if (!isZero(x(j)) && !isZero(z(i)))
      BOOST_LOG_TRIVIAL(error) << "Infeasible point given! Stay alert! " << x(j)
                               << " " << z(i) << " with i=" << i;
  };
  // std::stringstream enc_str;
  // for(auto vv:solEncoded) enc_str << vv <<" ";
  // BOOST_LOG_TRIVIAL (debug) << "Game::LCP::solEncode: Handling deviation with
  // encoding: "<< enc_str.str() << '\n';
  return solEncoded;
}

vector<short int> Game::LCP::solEncode(GRBModel *model) const
/// @brief Given a Gurobi model, extracts variable values and equation values,
/// encodes it in 0/+1/-1 format and returns it.
/// @warning Note that the vector returned by this function might have to be
/// explicitly deleted using the delete operator. For specific uses in
/// LCP::BranchAndPrune, this delete is handled by the class destructor.
{
  arma::vec x, z;
  if (!this->extractSols(model, z, x, true))
    return {}; // If infeasible model, return empty!
  else
    return this->solEncode(z, x);
}

LCP &Game::LCP::addPolyFromX(const arma::vec &x, bool &ret)
/**
 * Given a <i> feasible </i> point @p x, checks if any polyhedron that contains
 * @p x is already a part of this->Ai and this-> bi. If it is, then this does
 * nothing, except for printing a log message. If not, it adds a polyhedron
 * containing this vector.
 */
{
  const unsigned int nCompl = this->Compl.size();
  vector<short int> encoding = this->solEncode(x);
  std::stringstream enc_str;
  for (auto vv : encoding)
    enc_str << vv << " ";
  BOOST_LOG_TRIVIAL(trace)
      << "Game::LCP::addPolyFromX: Handling deviation with encoding: "
      << enc_str.str() << '\n';
  // Check if the encoding polyhedron is already in this->AllPolyhedra
  int found = -1;
  auto num_to_vec = [nCompl](unsigned int number) {
    std::vector<short int> binary{};
    for (unsigned int vv = 0; vv < nCompl; vv++) {
      binary.push_back(number % 2);
      number /= 2;
    }
    std::for_each(binary.begin(), binary.end(),
                  [](short int &vv) { vv = (vv == 0 ? -1 : 1); });
    std::reverse(binary.begin(), binary.end());
    return binary;
  }; // End of num_to_vec lambda definition

  for (const auto &i : AllPolyhedra) {
    std::vector<short int> bin = num_to_vec(i);
    if (encoding < bin) {
      BOOST_LOG_TRIVIAL(trace) << "LCP::addPolyFromX: Encoding " << i
                               << " already in All Polyhedra! ";
      ret = false;
      return *this;
    }
  }

  BOOST_LOG_TRIVIAL(trace)
      << "LCP::addPolyFromX: New encoding not in All Polyhedra! ";
  // If it is not in AllPolyhedra
  // First change any zero indices of encoding to 1
  for (short &i : encoding) {
    if (i == 0)
      ++i;
  }
  // And then add the relevant polyhedra
  ret = this->FixToPoly(encoding, false);
  // ret = true;
  return *this;
}

bool Game::LCP::FixToPoly(
    const vector<short int>
        Fix,        ///< A vector of +1 and -1 referring to which
                    ///< equations and variables are taking 0 value.
    bool checkFeas, ///< The polyhedron is added after ensuring feasibility, if
                    ///< this is true
    bool custom,    ///< Should the polyhedra be pushed into a custom vector of
                    ///< polyhedra as opposed to LCP::Ai and LCP::bi
    spmat_Vec *custAi, ///< If custom polyhedra vector is used, pointer to
                       ///< vector of LHS constraint matrix
    vec_Vec *custbi    /// If custom polyhedra vector is used, pointer
                       /// to vector of RHS of constraints
    )
/** @brief Computes the equation of the feasibility polyhedron corresponding to
 *the given @p Fix
 *	@details The computed polyhedron is always pushed into a vector of @p
 *arma::sp_mat and @p arma::vec If @p custom is false, this is the internal
 *attribute of LCP, which are LCP::Ai and LCP::bi. Otherwise, the vectors can be
 *provided as arguments.
 *	@p true value to @p checkFeas ensures that the polyhedron is pushed @e
 *only if it is feasible.
 * @returns @p true if successfully added, else false
 *	@warning Does not entertain 0 in the elements of *Fix. Only +1/-1 are
 *allowed to not encounter undefined behavior. As a result, not meant for high
 *level code. Instead use LCP::FixToPolies.
 */
{
  // std::vector representation to decimal number
  auto vec_to_num = [](std::vector<short int> binary) {
    unsigned int number = 0;
    unsigned int posn = 1;
    while (!binary.empty()) {
      short int bit = (binary.back() + 1) / 2; // The least significant bit
      number += (bit * posn);
      posn *= 2;         // Update place value
      binary.pop_back(); // Remove that bit
    }
    return number;
  };

  unsigned int FixNumber = vec_to_num(Fix);

  if (knownInfeas.find(FixNumber) != knownInfeas.end()) {
    BOOST_LOG_TRIVIAL(trace) << "Game::LCP::FixToPoly: Previously known "
                                "infeasible polyhedron. Not added"
                             << FixNumber;
    return false;
  }

  if (!custom && !AllPolyhedra.empty()) {
    if (AllPolyhedra.find(FixNumber) != AllPolyhedra.end()) {
      BOOST_LOG_TRIVIAL(trace)
          << "Game::LCP::FixToPoly: Previously added polyhedron. Not added "
          << FixNumber;
      return false;
    }
  }

  unique_ptr<arma::sp_mat> Aii =
      unique_ptr<arma::sp_mat>(new arma::sp_mat(nR, nC));
  Aii->zeros();
  unique_ptr<arma::vec> bii =
      unique_ptr<arma::vec>(new arma::vec(nR, arma::fill::zeros));
  for (unsigned int i = 0; i < this->nR; i++) {
    if (Fix.at(i) == 0) {
      throw string(
          "Error in Game::LCP::FixToPoly. 0s not allowed in argument vector");
    }
    if (Fix.at(i) == 1) // Equation to be fixed top zero
    {
      for (auto j = this->M.begin_row(i); j != this->M.end_row(i); ++j)
        if (!this->isZero((*j)))
          Aii->at(i, j.col()) =
              (*j); // Only mess with non-zero elements of a sparse matrix!
      bii->at(i) = -this->q(i);
    } else // Variable to be fixed to zero, i.e. x(j) <= 0 constraint to be
    // added
    {
      unsigned int varpos = (i >= this->LeadStart) ? i + this->nLeader : i;
      Aii->at(i, varpos) = 1;
      bii->at(i) = 0;
    }
  }
  bool add = !checkFeas;
  if (checkFeas) {
    unsigned int count{0};
    try {
      makeRelaxed();
      GRBModel model(this->RlxdModel);
      for (auto i : Fix) {
        if (i > 0)
          model.getVarByName("z_" + to_string(count)).set(GRB_DoubleAttr_UB, 0);
        if (i < 0)
          model
              .getVarByName("x_" + to_string(count >= this->LeadStart
                                                 ? count + nLeader
                                                 : count))
              .set(GRB_DoubleAttr_UB, 0);
        count++;
      }
      model.set(GRB_IntParam_OutputFlag, VERBOSE);
      model.optimize();
      if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL)
        add = true;
      else // Remember that this is an infeasible polyhedra
      {
        BOOST_LOG_TRIVIAL(trace)
            << "Game::LCP::FixToPoly: Detected infeasibility of " << FixNumber
            << " (GRB_STATUS=" << model.get(GRB_IntAttr_Status) << ")";
        knownInfeas.insert(FixNumber);
         notProcessed.erase(FixNumber);
      }
    } catch (const char *e) {
      cerr << "Error in Game::LCP::FixToPoly: " << e << '\n';
      throw;
    } catch (string e) {
      cerr << "String: Error in Game::LCP::FixToPoly: " << e << '\n';
      throw;
    } catch (exception &e) {
      cerr << "Exception: Error in Game::LCP::FixToPoly: " << e.what() << '\n';
      throw;
    } catch (GRBException &e) {
      cerr << "GRBException: Error in Game::LCP::FixToPoly: "
           << e.getErrorCode() << ": " << e.getMessage() << '\n';
      throw;
    }
  }
  if (add) {
    if (custom) {
      custAi->push_back(std::move(Aii));
      custbi->push_back(std::move(bii));
    } else {
      AllPolyhedra.insert(FixNumber);
      notProcessed.erase(FixNumber);
      this->Ai->push_back(std::move(Aii));
      this->bi->push_back(std::move(bii));
    }
    BOOST_LOG_TRIVIAL(debug)
        << "Game::LCP::FixToPoly:  Successfully added " << FixNumber;
    return true; // Successfully added
  }
  return false;
}

Game::LCP &Game::LCP::FixToPolies(
    const vector<short int>
        Fix,        ///< A vector of +1, 0 and -1 referring to which
                    ///< equations and variables are taking 0 value.
    bool checkFeas, ///< The polyhedron is added after ensuring feasibility, if
                    ///< this is true
    bool custom,    ///< Should the polyhedra be pushed into a custom vector of
                    ///< polyhedra as opposed to LCP::Ai and LCP::bi
    spmat_Vec *custAi, ///< If custom polyhedra vector is used, pointer to
                       ///< vector of LHS constraint matrix
    vec_Vec *custbi    /// If custom polyhedra vector is used, pointer
                       /// to vector of RHS of constraints
    )
/** @brief Computes the equation of the feasibility polyhedron corresponding to
 *the given @p Fix
 *	@details The computed polyhedron are always pushed into a vector of @p
 *arma::sp_mat and @p arma::vec If @p custom is false, this is the internal
 *attribute of LCP, which are LCP::Ai and LCP::bi. Otherwise, the vectors can be
 *provided as arguments.
 *	@p true value to @p checkFeas ensures that @e each polyhedron that is
 *pushed is feasible. not meant for high level code. Instead use
 *LCP::FixToPolies.
 *	@note A value of 0 in @p *Fix implies that polyhedron corresponding to
 *fixing the corresponding variable as well as the equation become candidates to
 *pushed into the vector. Hence this is preferred over LCP::FixToPoly for
 *high-level usage.
 */
{
  bool flag = false; // flag that there may be multiple polyhedra, i.e. 0 in
                     // some Fix entry
  vector<short int> MyFix(Fix);
  unsigned int i;
  for (i = 0; i < this->nR; i++) {
    if (Fix.at(i) == 0) {
      flag = true;
      break;
    }
  }
  if (flag) {
    MyFix[i] = 1;
    this->FixToPolies(MyFix, checkFeas, custom, custAi, custbi);
    MyFix[i] = -1;
    this->FixToPolies(MyFix, checkFeas, custom, custAi, custbi);
  } else
    this->FixToPoly(Fix, checkFeas, custom, custAi, custbi);
  return *this;
}

unsigned int Game::LCP::getNextPoly(Game::EPECAddPolyMethod method) const {
  /**
   * Returns a polyhedron (in its decimal encoding) that is neither already
   * known to be infeasible, nor already added in the inner approximation
   * representation.
   */
  switch (method) {
  case Game::EPECAddPolyMethod::sequential: {
    if (!notProcessed.empty())
      return *notProcessed.begin();
    else
      return maxTheoreticalPoly;
  } break;
  case Game::EPECAddPolyMethod::reverse_sequential: {
    if (!notProcessed.empty())
      return *--notProcessed.end();
    else
      return maxTheoreticalPoly;
  } break;
  case Game::EPECAddPolyMethod::random: {
    std::random_device random_device;
    std::mt19937 engine{random_device()};
    std::uniform_int_distribution<int> dist(0, this->notProcessed.size() - 1);
    return *(std::next(this->notProcessed.begin(), dist(engine)));
  }
  default: {
    BOOST_LOG_TRIVIAL(error)
        << "Error in Game::LCP::getNextPoly: unrecognized method "
        << to_string(method);
    throw;
  }
  }
}

std::set<std::vector<short int>>
Game::LCP::addAPoly(unsigned int nPoly, Game::EPECAddPolyMethod method,
                    std::set<std::vector<short int>> Polys) {
  /**
   * Tries to add at most @p nPoly number of polyhedra to the inner
   * approximation representation of the current LCP. The set of added polyhedra
   * (+1/-1 encoding) is appended to  @p Polys and returned. The only reason
   * fewer polyhedra might be added is that the fewer polyhedra already
   * represent the feasible region of the LCP.
   * @p method is casted from Game::EPEC::EPECAddPolyMethod
   */

  // We already have polyhedrain AllPolyhedra and polyhedra in
  // knownInfeas, that are known to be infeasible.
  // Effective maximum of number of polyhedra that can be added
  // at most
  const unsigned int nCompl = this->Compl.size();
  const unsigned int possibleMaxPoly =
      maxTheoreticalPoly - AllPolyhedra.size() - knownInfeas.size();

  if (maxTheoreticalPoly < nPoly) { // If you cannot add that many polyhedra
    BOOST_LOG_TRIVIAL(warning)      // Then issue a warning
        << "Warning in Game::LCP::randomPoly: "
        << "Cannot add " << nPoly << " polyhedra. Promising a maximum of "
        << maxTheoreticalPoly;
    nPoly = maxTheoreticalPoly; // and update maximum possibly addable
  }

  if (nPoly == 0) // If nothing to be added, then nothing to be done
    return Polys;

  if (nPoly < 0) // There is no way that this can happen!
  {
    BOOST_LOG_TRIVIAL(error) << "nPoly can't be negative, i.e., " << nPoly;
    throw string(
        "Error in Game::LCP::addAPoly: nPoly reached a negative value!");
  }

  // Otherwise try adding one polyhedron
  unsigned int choice_decimal = this->getNextPoly(method);
  cout << "adding" << choice_decimal;
  if (choice_decimal >= maxTheoreticalPoly)
    return Polys;

  // Now convert choice_decimal to binary vector representation
  auto num_to_vec = [nCompl](unsigned int number) {
    std::vector<short int> binary{};
    for (unsigned int vv = 0; vv < nCompl; vv++) {
      binary.push_back(number % 2);
      number /= 2;
    }
    std::for_each(binary.begin(), binary.end(),
                  [](short int &vv) { vv = (vv == 0 ? -1 : 1); });
    std::reverse(binary.begin(), binary.end());
    return binary;
  }; // End of num_to_vec lambda definition

  const std::vector<short int> choice = num_to_vec(choice_decimal);

  auto added = this->FixToPoly(choice, true);
  if (added) // If choice is added to All Polyhedra
  {
    Polys.insert(choice); // Add it to set of added polyhedra
    return this->addAPoly(nPoly - 1, method,
                          Polys); // Now we have one less polyhedron to add
  } else {
    return this->addAPoly(
        nPoly, method,
        Polys); // We have to add the same number of polyhedra anyway :(
  }
}

Game::LCP &Game::LCP::EnumerateAll(
    const bool
        solveLP ///< Should the poyhedra added be checked for feasibility?
    )
/**
 * @brief Brute force computation of LCP feasible region
 * @details Computes all @f$2^n@f$ polyhedra defining the LCP feasible region.
 * Th ese are always added to LCP::Ai and LCP::bi
 */
{
  vector<short int> Fix = vector<short int>(nR, 0);
  this->Ai->clear();
  this->bi->clear();
  this->FixToPolies(Fix, solveLP);
  if (this->Ai->empty()) {
    BOOST_LOG_TRIVIAL(warning)
        << "Empty vector of polyhedra given! Problem might be infeasible."
        << '\n';
    // 0 <= -1 for infeasability
    unique_ptr<arma::sp_mat> A(new arma::sp_mat(1, this->M.n_cols));
    unique_ptr<arma::vec> b(new arma::vec(1));
    b->at(0) = -1;
    this->Ai->push_back(std::move(A));
    this->bi->push_back(std::move(b));
  }
  return *this;
}

Game::LCP &Game::LCP::makeQP(
    Game::QP_objective
        &QP_obj, ///< The objective function of the QP to be returned. @warning
                 ///< Size of this parameter might change!
    Game::QP_Param &QP ///< The output parameter where the final Game::QP_Param
                       ///< object is stored

) {
  // Original sizes
  const unsigned int Nx_old{static_cast<unsigned int>(QP_obj.C.n_cols)};

  Game::QP_constraints QP_cons;
  this->feasiblePolyhedra = this->ConvexHull(QP_cons.B, QP_cons.b);
  BOOST_LOG_TRIVIAL(debug) << "LCP::makeQP: No. feasible polyhedra: "
                           << this->feasiblePolyhedra;
  // Updated size after convex hull has been computed.
  const unsigned int Ncons{static_cast<unsigned int>(QP_cons.B.n_rows)};
  const unsigned int Ny{static_cast<unsigned int>(QP_cons.B.n_cols)};
  // Resizing entities.
  QP_cons.A.zeros(Ncons, Nx_old);
  QP_obj.c = resize_patch(QP_obj.c, Ny, 1);
  QP_obj.C = resize_patch(QP_obj.C, Ny, Nx_old);
  QP_obj.Q = resize_patch(QP_obj.Q, Ny, Ny);
  // Setting the QP_Param object
  QP.set(QP_obj, QP_cons);
  return *this;
}

unique_ptr<GRBModel> Game::LCP::LCPasQP(bool solve)
/** @brief Solves the LCP as a QP using Gurobi */
/** Removes all complementarity constraints from the QP's constraints. Instead,
 * the sum of products of complementarity pairs is minimized. If the optimal
 * value turns out to be 0, then it is actually a solution of the LCP. Else the
 * LCP is infeasible.
 * @warning Solves the LCP feasibility problem. Not the MPEC optimization
 * problem.
 * */
{
  this->makeRelaxed();
  unique_ptr<GRBModel> model(new GRBModel(this->RlxdModel));
  GRBQuadExpr obj = 0;
  GRBVar x[this->nR];
  GRBVar z[this->nR];
  for (const auto p : this->Compl) {
    unsigned int i = p.first;
    unsigned int j = p.second;
    z[i] = model->getVarByName("z_" + to_string(i));
    x[i] = model->getVarByName("x_" + to_string(j));
    obj += x[i] * z[i];
  }
  model->setObjective(obj, GRB_MINIMIZE);
  if (solve) {
    try {
      model->optimize();
      int status = model->get(GRB_IntAttr_Status);
      if (status != GRB_OPTIMAL ||
          model->get(GRB_DoubleAttr_ObjVal) > this->eps)
        throw "LCP infeasible";
    } catch (const char *e) {
      cerr << "Error in Game::LCP::LCPasQP: " << e << '\n';
      throw;
    } catch (string e) {
      cerr << "String: Error in Game::LCP::LCPasQP: " << e << '\n';
      throw;
    } catch (exception &e) {
      cerr << "Exception: Error in Game::LCP::LCPasQP: " << e.what() << '\n';
      throw;
    } catch (GRBException &e) {
      cerr << "GRBException: Error in Game::LCP::LCPasQP: " << e.getErrorCode()
           << "; " << e.getMessage() << '\n';
      throw;
    }
  }
  return model;
}

unique_ptr<GRBModel> Game::LCP::LCPasMIP(bool solve)
/**
 * @brief Helps solving an LCP as an MIP using bigM constraints
 * @returns A unique_ptr to GRBModel that has the equivalent MIP
 * @details The MIP problem that is returned by this function is equivalent to
 * the LCP problem provided the value of bigM is large enough.
 * @note This solves just the feasibility problem. Should you need  a leader's
 * objective function, use LCP::MPECasMILP or LCP::MPECasMIQP
 */
{
  return this->LCPasMIP({}, {}, solve);
}

unique_ptr<GRBModel>
Game::LCP::MPECasMILP(const arma::sp_mat &C, const arma::vec &c,
                      const arma::vec &x_minus_i, bool solve)
/**
 * @brief Helps solving an LCP as an MIP.
 * @returns A unique_ptr to GRBModel that has the equivalent MIP
 * @details The MIP problem that is returned by this function is equivalent to
 * the LCP problem. The function
 * differs from LCP::LCPasMIP by the fact that, this explicitly takes a leader
 * objective, and returns an object with this objective.
 * @note The leader's objective has to be linear here. For quadratic objectives,
 * refer LCP::MPECasMIQP
 */
{
  unique_ptr<GRBModel> model = this->LCPasMIP(true);
  // Reset the solution limit. We need to solve to optimality
  model->set(GRB_IntParam_SolutionLimit, GRB_MAXINT);
  if (C.n_cols != x_minus_i.n_rows)
    throw string("Bad size of x_minus_i");
  if (c.n_rows != C.n_rows)
    throw string("Bad size of c");
  arma::vec Cx(c.n_rows, arma::fill::zeros);
  try {
    Cx = C * x_minus_i;
  } catch (exception &e) {
    cerr << "Exception in Game::LCP::MPECasMIQP: " << e.what() << '\n';
    throw;
  } catch (string &e) {
    cerr << "Exception in Game::LCP::MPECasMIQP: " << e << '\n';
    throw;
  }
  arma::vec obj = c + Cx;
  GRBLinExpr expr{0};
  for (unsigned int i = 0; i < obj.n_rows; i++)
    expr += obj.at(i) * model->getVarByName("x_" + to_string(i));
  model->setObjective(expr, GRB_MINIMIZE);
  model->set(GRB_IntParam_OutputFlag, VERBOSE);
  if (solve)
    model->optimize();
  return model;
}

unique_ptr<GRBModel>
Game::LCP::MPECasMIQP(const arma::sp_mat &Q, const arma::sp_mat &C,
                      const arma::vec &c, const arma::vec &x_minus_i,
                      bool solve)
/**
 * @brief Helps solving an LCP as an MIQPs.
 * @returns A unique_ptr to GRBModel that has the equivalent MIQP
 * @details The MIQP problem that is returned by this function is equivalent to
 * the LCP problem provided the value of bigM is large enough. The function
 * differs from LCP::LCPasMIP by the fact that, this explicitly takes a leader
 * objective, and returns an object with this objective. This allows quadratic
 * leader objective. If you are aware that the leader's objective is linear, use
 * the faster method LCP::MPECasMILP
 */
{
  auto model = this->MPECasMILP(C, c, x_minus_i, false);
  /// Note that if the matrix Q is a zero matrix, then this returns a Gurobi
  /// MILP model as opposed to MIQP model. This enables Gurobi to use its much
  /// advanced MIP solver
  if (Q.n_nonzero != 0) // If Q is zero, then just solve MIP as opposed to MIQP!
  {
    GRBLinExpr linexpr = model->getObjective(0);
    GRBQuadExpr expr{linexpr};
    for (auto it = Q.begin(); it != Q.end(); ++it)
      expr += (*it) * model->getVarByName("x_" + to_string(it.row())) *
              model->getVarByName("x_" + to_string(it.col()));
    model->setObjective(expr, GRB_MINIMIZE);
  }
  if (solve)
    model->optimize();
  return model;
}

void Game::LCP::write(string filename, bool append) const {
  ofstream outfile(filename, append ? ios::app : ios::out);

  outfile << nR << " rows and " << nC << " columns in the LCP\n";
  outfile << "LeadStart: " << LeadStart << " \nLeadEnd: " << LeadEnd
          << " \nnLeader: " << nLeader << "\n\n";

  outfile << "M: " << this->M;
  outfile << "q: " << this->q;
  outfile << "Complementarity: \n";
  for (const auto &p : this->Compl)
    outfile << "<" << p.first << ", " << p.second << ">"
            << "\t";
  outfile << "A: " << this->_A;
  outfile << "b: " << this->_b;
  outfile.close();
}

void Game::LCP::save(string filename, bool erase) const {
  Utils::appendSave(string("LCP"), filename, erase);
  Utils::appendSave(this->M, filename, string("LCP::M"), false);
  Utils::appendSave(this->q, filename, string("LCP::q"), false);

  Utils::appendSave(this->LeadStart, filename, string("LCP::LeadStart"), false);
  Utils::appendSave(this->LeadEnd, filename, string("LCP::LeadEnd"), false);

  Utils::appendSave(this->_A, filename, string("LCP::_A"), false);
  Utils::appendSave(this->_b, filename, string("LCP::_b"), false);

  BOOST_LOG_TRIVIAL(trace) << "Saved LCP to file " << filename;
}

long int Game::LCP::load(string filename, long int pos) {
  if (!this->env)
    throw string("Error in LCP::load: To load LCP from file, it has to be "
                 "constructed using LCP(GRBEnv*) constructor");

  string headercheck;
  pos = Utils::appendRead(headercheck, filename, pos);
  if (headercheck != "LCP")
    throw string("Error in LCP::load: In valid header - ") + headercheck;

  arma::sp_mat M, A;
  arma::vec q, b;
  unsigned int LeadStart, LeadEnd;
  pos = Utils::appendRead(M, filename, pos, string("LCP::M"));
  pos = Utils::appendRead(q, filename, pos, string("LCP::q"));
  pos = Utils::appendRead(LeadStart, filename, pos, string("LCP::LeadStart"));
  pos = Utils::appendRead(LeadEnd, filename, pos, string("LCP::LeadEnd"));
  pos = Utils::appendRead(A, filename, pos, string("LCP::_A"));
  pos = Utils::appendRead(b, filename, pos, string("LCP::_b"));

  this->M = M;
  this->q = q;
  this->_A = A;
  this->_b = b;
  defConst(env);
  this->LeadStart = LeadStart;
  this->LeadEnd = LeadEnd;

  this->nLeader = this->LeadEnd - this->LeadStart + 1;
  this->nLeader = this->nLeader > 0 ? this->nLeader : 0;
  for (unsigned int i = 0; i < M.n_rows; i++) {
    unsigned int count = i < LeadStart ? i : i + nLeader;
    Compl.push_back({i, count});
  }
  std::sort(Compl.begin(), Compl.end(),
            [](std::pair<unsigned int, unsigned int> a,
               std::pair<unsigned int, unsigned int> b) {
              return a.first <= b.first;
            });
  this->initializeNotPorcessed();
  return pos;
}

std::string Game::LCP::feas_detail_str() const {
  std::stringstream ss;
  ss << "\tProven feasible: ";
  for (auto vv : this->AllPolyhedra)
    ss << vv << ' ';
  // ss << "\tProven infeasible: ";
  // for (auto vv : this->knownInfeas)
  // ss << vv << ' ';

  return ss.str();
}
