#include "PolyLCP.h"
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

unsigned int Game::polyLCP::ConvexHull(
    arma::sp_mat &A, ///< Convex hull inequality description
                     ///< LHS to be stored here
    arma::vec &b)    ///< Convex hull inequality description RHS
///< to be stored here
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
}
Game::polyLCP &Game::polyLCP::addPolyFromX(const arma::vec &x, bool &ret)
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
      << "Game::polyLCP::addPolyFromX: Handling deviation with encoding: "
      << enc_str.str() << '\n';
  // Check if the encoding polyhedron is already in this->AllPolyhedra
  for (const auto &i : AllPolyhedra) {
    std::vector<short int> bin = num_to_vec(i, nCompl);
    if (encoding < bin) {
      BOOST_LOG_TRIVIAL(trace) << "Game::polyLCP::addPolyFromX: Encoding " << i
                               << " already in All Polyhedra! ";
      ret = false;
      return *this;
    }
  }

  BOOST_LOG_TRIVIAL(trace)
      << "Game::polyLCP::addPolyFromX: New encoding not in All Polyhedra! ";
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

bool Game::polyLCP::FixToPoly(
    const vector<short int> Fix, ///< A vector of +1 and -1 referring to which
    ///< equations and variables are taking 0 value.
    bool checkFeas, ///< The polyhedron is added after ensuring feasibility, if
    ///< this is true
    bool custom, ///< Should the polyhedra be pushed into a custom vector of
    ///< polyhedra as opposed to LCP::Ai and LCP::bi
    spmat_Vec *custAi, ///< If custom polyhedra vector is used, pointer to
    ///< vector of LHS constraint matrix
    vec_Vec *custbi /// If custom polyhedra vector is used, pointer
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
  unsigned int FixNumber = vec_to_num(Fix);
  BOOST_LOG_TRIVIAL(trace)
      << "Game::polyLCP::FixToPoly: Working on polyhedron #" << FixNumber;

  if (knownInfeas.find(FixNumber) != knownInfeas.end()) {
    BOOST_LOG_TRIVIAL(trace) << "Game::polyLCP::FixToPoly: Previously known "
                                "infeasible polyhedron #"
                             << FixNumber;
    return false;
  }

  if (!custom && !AllPolyhedra.empty()) {
    if (AllPolyhedra.find(FixNumber) != AllPolyhedra.end()) {
      BOOST_LOG_TRIVIAL(trace)
          << "Game::polyLCP::FixToPoly: Previously added polyhedron #"
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
      throw string("Error in Game::polyLCP::FixToPoly. 0s not allowed in "
                   "argument vector");
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
    add = this->checkPolyFeas(Fix);
  }
  if (add) {
    if (custom) {
      custAi->push_back(std::move(Aii));
      custbi->push_back(std::move(bii));
    } else {
      AllPolyhedra.insert(FixNumber);
      this->Ai->push_back(std::move(Aii));
      this->bi->push_back(std::move(bii));
    }
    return true; // Successfully added
  }
  return false;
}

Game::polyLCP &Game::polyLCP::FixToPolies(
    const vector<short int>
        Fix, ///< A vector of +1, 0 and -1 referring to which
    ///< equations and variables are taking 0 value.
    bool checkFeas, ///< The polyhedron is added after ensuring feasibility, if
    ///< this is true
    bool custom, ///< Should the polyhedra be pushed into a custom vector of
    ///< polyhedra as opposed to LCP::Ai and LCP::bi
    spmat_Vec *custAi, ///< If custom polyhedra vector is used, pointer to
    ///< vector of LHS constraint matrix
    vec_Vec *custbi /// If custom polyhedra vector is used, pointer
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

unsigned long int Game::polyLCP::getNextPoly(Game::EPECAddPolyMethod method) {
  /**
   * Returns a polyhedron (in its decimal encoding) that is neither already
   * known to be infeasible, nor already added in the inner approximation
   * representation.
   */

  switch (method) {
  case Game::EPECAddPolyMethod::sequential: {
    while (this->sequentialPolyCounter < this->maxTheoreticalPoly) {
      const bool isAll =
          AllPolyhedra.find(this->sequentialPolyCounter) != AllPolyhedra.end();
      const bool isInfeas =
          knownInfeas.find(this->sequentialPolyCounter) != knownInfeas.end();
      this->sequentialPolyCounter++;
      if (!isAll && !isInfeas) {
        return this->sequentialPolyCounter - 1;
      }
    }
    return this->maxTheoreticalPoly;
  } break;
  case Game::EPECAddPolyMethod::reverse_sequential: {
    while (this->reverseSequentialPolyCounter >= 0) {
      const bool isAll =
          AllPolyhedra.find(this->reverseSequentialPolyCounter) !=
          AllPolyhedra.end();
      const bool isInfeas =
          knownInfeas.find(this->reverseSequentialPolyCounter) !=
          knownInfeas.end();
      this->reverseSequentialPolyCounter--;
      if (!isAll && !isInfeas) {
        return this->reverseSequentialPolyCounter + 1;
      }
    }
    return this->maxTheoreticalPoly;
  } break;
  case Game::EPECAddPolyMethod::random: {
    static std::mt19937 engine{this->addPolyMethodSeed};
    std::uniform_int_distribution<unsigned long int> dist(
        0, this->maxTheoreticalPoly - 1);
    if ((knownInfeas.size() + AllPolyhedra.size()) == this->maxTheoreticalPoly)
      return this->maxTheoreticalPoly;
    while (true) {
      unsigned long int randomPolyId = dist(engine);
      const bool isAll = AllPolyhedra.find(randomPolyId) != AllPolyhedra.end();
      const bool isInfeas = knownInfeas.find(randomPolyId) != knownInfeas.end();
      if (!isAll && !isInfeas)
        return randomPolyId;
    }
  }
  default: {
    BOOST_LOG_TRIVIAL(error)
        << "Error in Game::polyLCP::getNextPoly: unrecognized method "
        << to_string(method);
    throw;
  }
  }
}

std::set<std::vector<short int>>
Game::polyLCP::addAPoly(unsigned long int nPoly, Game::EPECAddPolyMethod method,
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

  if (this->maxTheoreticalPoly <
      nPoly) {                 // If you cannot add that many polyhedra
    BOOST_LOG_TRIVIAL(warning) // Then issue a warning
        << "Warning in Game::polyLCP::randomPoly: "
        << "Cannot add " << nPoly << " polyhedra. Promising a maximum of "
        << this->maxTheoreticalPoly;
    nPoly = this->maxTheoreticalPoly; // and update maximum possibly addable
  }

  if (nPoly == 0) // If nothing to be added, then nothing to be done
    return Polys;

  if (nPoly < 0) // There is no way that this can happen!
  {
    BOOST_LOG_TRIVIAL(error) << "nPoly can't be negative, i.e., " << nPoly;
    throw string(
        "Error in Game::polyLCP::addAPoly: nPoly reached a negative value!");
  }

  bool complete{false};
  while (!complete) {
    unsigned long int choice_decimal = this->getNextPoly(method);
    if (choice_decimal >= this->maxTheoreticalPoly)
      return Polys;

    const std::vector<short int> choice = num_to_vec(choice_decimal, nCompl);
    auto added = this->FixToPoly(choice, true);
    if (added) // If choice is added to All Polyhedra
    {
      Polys.insert(choice); // Add it to set of added polyhedra
      if (Polys.size() == nPoly) {
        complete = true;
        return Polys;
      }
    }
  }
  return Polys;
}
bool Game::polyLCP::addThePoly(const unsigned long int &decimalEncoding) {
  if (this->maxTheoreticalPoly < decimalEncoding) {
    // This polyhedron does not exist
    BOOST_LOG_TRIVIAL(warning)
        << "Warning in Game::polyLCP::addThePoly: Cannot add "
        << decimalEncoding << " polyhedra, since it does not exist!";
    return false;
  }
  const unsigned int nCompl = this->Compl.size();
  const std::vector<short int> choice = num_to_vec(decimalEncoding, nCompl);
  return this->FixToPoly(choice, true);
}

Game::polyLCP &Game::polyLCP::EnumerateAll(
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

void Game::polyLCP::makeQP(
    Game::QP_objective
        &QP_obj, ///< The objective function of the QP to be returned. @warning
    ///< Size of this parameter might change!
    Game::QP_Param &QP ///< The output parameter where the final Game::QP_Param
                       ///< object is stored

) {
  // Original sizes
  if (this->Ai->empty())
    return;
  const unsigned int Nx_old{static_cast<unsigned int>(QP_obj.C.n_cols)};

  Game::QP_constraints QP_cons;
  this->feasiblePolyhedra = this->ConvexHull(QP_cons.B, QP_cons.b);
  BOOST_LOG_TRIVIAL(trace) << "PolyLCP::makeQP: No. feasible polyhedra: "
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
}

std::string Game::polyLCP::feas_detail_str() const {
  std::stringstream ss;
  ss << "\tProven feasible: ";
  for (auto vv : this->AllPolyhedra)
    ss << vv << ' ';
  // ss << "\tProven infeasible: ";
  // for (auto vv : this->knownInfeas)
  // ss << vv << ' ';

  return ss.str();
}

unsigned int Game::polyLCP::conv_Npoly() const {
  /**
   * To be used in interaction with Game::LCP::ConvexHull.
   * Gives the number of polyhedra in the current inner approximation of the LCP
   * feasible region.
   */
  return this->AllPolyhedra.size();
}

unsigned int Game::polyLCP::conv_PolyPosition(const unsigned long int i) const {
  /**
   * For the convex hull of the LCP feasible region computed, a bunch of
   * variables are added for extended formulation and the added variables c
   */
  const unsigned int nPoly = this->conv_Npoly();
  if (i > nPoly) {
    BOOST_LOG_TRIVIAL(error) << "Error in Game::polyLCP::conv_PolyPosition: "
                                "Invalid argument. Out of bounds for i";
    throw std::string("Error in Game::polyLCP::conv_PolyPosition: Invalid "
                      "argument. Out of bounds for i");
  }
  const unsigned int nC = this->M.n_cols;
  return nC + i * nC;
}

unsigned int Game::polyLCP::conv_PolyWt(const unsigned long int i) const {
  /**
   * To be used in interaction with Game::LCP::ConvexHull.
   * Gives the position of the variable, which assigns the convex weight to the
   * i-th polyhedron.
   *
   * However, if the inner approximation has exactly one polyhedron,
   * then returns 0.
   */
  const unsigned int nPoly = this->conv_Npoly();
  if (nPoly <= 1) {
    return 0;
  }
  if (i > nPoly) {
    throw std::string("Error in Game::polyLCP::conv_PolyWt: "
                      "Invalid argument. Out of bounds for i");
  }
  const unsigned int nC = this->M.n_cols;

  return nC + nPoly * nC + i;
}

bool Game::polyLCP::checkPolyFeas(
    const unsigned long int
        &decimalEncoding ///< Decimal encoding for the polyhedron
) {
  return this->checkPolyFeas(num_to_vec(decimalEncoding, this->Compl.size()));
}

bool Game::polyLCP::checkPolyFeas(
    const vector<short int> &Fix ///< A vector of +1 and -1 referring to which
    ///< equations and variables are taking 0 value.)
) {

  unsigned long int FixNumber = vec_to_num(Fix);
  if (knownInfeas.find(FixNumber) != knownInfeas.end()) {
    BOOST_LOG_TRIVIAL(trace)
        << "Game::polyLCP::checkPolyFeas: Previously known "
           "infeasible polyhedron. "
        << FixNumber;
    return false;
  }

  if (feasiblePoly.find(FixNumber) != feasiblePoly.end()) {
    BOOST_LOG_TRIVIAL(trace)
        << "Game::polyLCP::checkPolyFeas: Previously known "
           "feasible polyhedron."
        << FixNumber;
    return true;
  }

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
    if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
      feasiblePoly.insert(FixNumber);
      return true;
    } else {
      BOOST_LOG_TRIVIAL(trace)
          << "Game::polyLCP::checkPolyFeas: Detected infeasibility of "
          << FixNumber << " (GRB_STATUS=" << model.get(GRB_IntAttr_Status)
          << ")";
      knownInfeas.insert(FixNumber);
      return false;
    }
  } catch (const char *e) {
    cerr << "Error in Game::polyLCP::checkPolyFeas: " << e << '\n';
    throw;
  } catch (string e) {
    cerr << "String: Error in Game::polyLCP::checkPolyFeas: " << e << '\n';
    throw;
  } catch (exception &e) {
    cerr << "Exception: Error in Game::polyLCP::checkPolyFeas: " << e.what()
         << '\n';
    throw;
  } catch (GRBException &e) {
    cerr << "GRBException: Error in Game::polyLCP::checkPolyFeas: "
         << e.getErrorCode() << ": " << e.getMessage() << '\n';
    throw;
  }
  return false;
}
