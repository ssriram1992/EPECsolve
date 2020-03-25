#include "LCP.h"
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



void Game::LCP::defConst(GRBEnv *env)
/**
 * @brief Assign default values to LCP attributes
 * @details Internal member that can be called from multiple constructors
 * to assign default values to some attributes of the class.
 */
{
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
}

Game::LCP::LCP(GRBEnv *env, const NashGame &N)
    : RlxdModel(*env)
/**
 *	@brief Constructor given a NashGame
 *	@details Given a NashGame, computes the KKT of the lower levels, and
 *makes the appropriate LCP object.
 *
 *	This constructor is the most suited for high-level usage.
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
        throw string("Game::LCP::makeRelaxed: A and b are incompatible! Thrown "
                     "from makeRelaxed()");
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
    throw string(
        "Game::LCP::LCPasMIP: Bad size for Fixes in Game::LCP::LCPasMIP");
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
  const unsigned int nR_t = M.n_rows;
  const unsigned int nC_t = M.n_cols;
  if (throwErr) {
    if (nR_t != q.n_rows)
      throw "Game::LCP::errorCheck: M and q have unequal number of rows";
    if (nR_t + nLeader != nC)
      throw "Game::LCP::errorCheck: Inconsistency between number of leader "
            "vars " +
          to_string(nLeader) + ", number of rows " + to_string(nR_t) +
          " and number of cols " + to_string(nC);
  }
  return (nR_t == q.n_rows && nR_t + nLeader == nC_t);
}

void Game::LCP::print(string end) {
  cout << "LCP with " << this->nR << " rows and " << this->nC << " columns."
       << end;
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
        throw "Game::LCP::LCPasQP: LCP infeasible";
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
    throw string("Game::LCP::LCPasMILP: Bad size of x_minus_i");
  if (c.n_rows != C.n_rows)
    throw string("Game::LCP::LCPasMILP: Bad size of c");
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

  arma::sp_mat M_t, A;
  arma::vec q_t, b;
  unsigned int LeadStart_t, LeadEnd_t;
  pos = Utils::appendRead(M_t, filename, pos, string("LCP::M"));
  pos = Utils::appendRead(q_t, filename, pos, string("LCP::q"));
  pos = Utils::appendRead(LeadStart_t, filename, pos, string("LCP::LeadStart"));
  pos = Utils::appendRead(LeadEnd_t, filename, pos, string("LCP::LeadEnd"));
  pos = Utils::appendRead(A, filename, pos, string("LCP::_A"));
  pos = Utils::appendRead(b, filename, pos, string("LCP::_b"));

  this->M = M_t;
  this->q = q_t;
  this->_A = A;
  this->_b = b;
  defConst(env);
  this->LeadStart = LeadStart_t;
  this->LeadEnd = LeadEnd_t;

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
  return pos;
}
