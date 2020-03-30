#include "OuterLCP.h"
#include <boost/log/trivial.hpp>

using namespace std;
using namespace Utils;
void outerLCP::makeQP(Game::QP_objective &QP_obj, Game::QP_Param &QP) {}

void outerLCP::outerApproximate(const std::vector<bool> Encoding) {
  if (Encoding.size() != this->Compl.size()) {
    BOOST_LOG_TRIVIAL(error)
        << "Game::outerLCP::outerApproximate: wrong encoding size";
    throw;
  }
  vector<short int> Fix;
  for (bool i : Encoding) {
    if (i)
      Fix.push_back(2);
    else
      Fix.push_back(0);
  }
  this->buildComponents(Fix);
}
void outerLCP::buildComponents(const std::vector<short int> Fix) {
  vector<short int> MyFix(Fix);
  unsigned int i;
  bool flag = false;
  for (i = 0; i < this->nR; i++) {
    if (Fix.at(i) == 2) {
      flag = true;
      break;
    }
  }
  if (flag) {
    MyFix[i] = 1;
    this->buildComponents(MyFix);
    MyFix[i] = -1;
    this->buildComponents(MyFix);
  } else
    this->addComponent(Fix, true);
}

bool Game::outerLCP::addComponent(
    const vector<short int>
        fix, ///< A vector of +1,-1 and zeros referring to which
    ///< equations and variables are taking 0 value. +1 means equation set to
    ///< zero, -1 variable, and zero  none of the two
    bool checkFeas, ///< The component is added after ensuring feasibility, if
    ///< this is true
    bool custom, ///< Should the components be pushed into a custom vector of
    ///< polyhedra as opposed to outerLCP::Ai and outerLCP::bi
    spmat_Vec *custAi, ///< If custom polyhedra vector is used, pointer to the
                       ///< LHS matrix
    vec_Vec *custbi ///< If custom polyhedra vector is used, pointer to the RHS
                    ///< vector
) {
  unsigned int fixNumber = vec_to_num(fix);
  BOOST_LOG_TRIVIAL(trace)
      << "Game::outerLCP::addComponent: Working on polyhedron #" << fixNumber;
  bool eval;
  if (checkFeas)
    eval = this->checkComponentFeas(fix);
  else
    eval = true;

  if (eval) {
    if (!custom && !Approximation.empty()) {
      if (Approximation.find(fixNumber) != Approximation.end()) {
        BOOST_LOG_TRIVIAL(trace)
            << "Game::outerLCP::addComponent: Previously added polyhedron #"
            << fixNumber;
        return false;
      }
    }
    unique_ptr<arma::sp_mat> Aii =
        unique_ptr<arma::sp_mat>(new arma::sp_mat(nR, nC));
    Aii->zeros();
    unique_ptr<arma::vec> bii =
        unique_ptr<arma::vec>(new arma::vec(nR, arma::fill::zeros));
    for (unsigned int i = 0; i < this->nR; i++) {
      if (fix.at(i) == 1) // Equation to be fixed top zero
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
    if (custom) {
      custAi->push_back(std::move(Aii));
      custbi->push_back(std::move(bii));
    } else {
      Approximation.insert(fixNumber);
      this->Ai->push_back(std::move(Aii));
      this->bi->push_back(std::move(bii));
    }
    return true; // Successfully added
  }
  BOOST_LOG_TRIVIAL(trace)
      << "Game::outerLCP::addComponent: Checkfeas + Infeasible polyhedron #"
      << fixNumber;
  return false;
}

bool outerLCP::checkComponentFeas(const std::vector<short int> &Fix) {
  unsigned long int fixNumber = vec_to_num(Fix);
  if (knownInfeas.find(fixNumber) != knownInfeas.end()) {
    BOOST_LOG_TRIVIAL(trace)
        << "Game::outerLCP::checkComponentFeas: Previously known "
           "infeasible component #"
        << fixNumber;
    return false;
  }

  if (knownFeasible.find(fixNumber) != knownFeasible.end()) {
    BOOST_LOG_TRIVIAL(trace)
        << "Game::outerLCP::checkComponentFeas: Previously known "
           "feasible polyhedron #"
        << fixNumber;
    return true;
  }
  for (auto element : knownInfeas) {
    if (this->isParent(num_to_vec(element, this->Compl.size()), Fix)) {
      BOOST_LOG_TRIVIAL(trace)
          << "Game::outerLCP::checkComponentFeas: #" << fixNumber
          << " is a child "
             "of an infeasible polyhedron";
      return true;
    }
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
      knownFeasible.insert(fixNumber);
      return true;
    } else {
      BOOST_LOG_TRIVIAL(trace)
          << "Game::outerLCP::checkComponentFeas: Detected infeasibility of #"
          << fixNumber << " (GRB_STATUS=" << model.get(GRB_IntAttr_Status)
          << ")";
      knownInfeas.insert(fixNumber);
      return false;
    }
  } catch (const char *e) {
    cerr << "Error in Game::outerLCP::checkComponentFeas: " << e << '\n';
    throw;
  } catch (string e) {
    cerr << "String: Error in Game::outerLCP::checkComponentFeas: " << e
         << '\n';
    throw;
  } catch (exception &e) {
    cerr << "Exception: Error in Game::outerLCP::checkComponentFeas: "
         << e.what() << '\n';
    throw;
  } catch (GRBException &e) {
    cerr << "GRBException: Error in Game::outerLCP::checkComponentFeas: "
         << e.getErrorCode() << ": " << e.getMessage() << '\n';
    throw;
  }
  return false;
}

bool outerLCP::isParent(const vector<short int> &father,
                        const vector<short int> &child) {
  for (int i = 0; i < father.size(); ++i) {
    if (father.at(i) != 0) {
      if (child.at(i) != father.at(i))
        return false;
    }
  }
  return true;
}