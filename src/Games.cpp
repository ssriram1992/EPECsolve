#include "games.h"
#include <algorithm>
#include <armadillo>
#include <array>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>
#include <memory>

using namespace std;
using namespace Utils;

bool Game::isZero(arma::mat M, double tol) noexcept {
  /**
   * @brief
   * Checking if a given matrix M is a zero matrix
   *
   * @param tol Tolerance, below which a number is treated as 0
   * @warning tol < 0 always returns @p false with no error.
   *
   */
  return (arma::min(arma::min(abs(M))) <= tol);
}

bool Game::isZero(arma::sp_mat M, double tol) noexcept {
  /**
   * @brief
   * Checking if a given sparse matrix M is a zero matrix
   *
   * @param tol Tolerance, below which a number is treated as 0
   *
   */
  if (M.n_nonzero == 0)
    return true;
  return (arma::min(arma::min(abs(M))) <= tol);
}

void Game::print(const perps &C) noexcept {
  for (auto p : C)
    cout << "<" << p.first << ", " << p.second << ">"
         << "\t";
}

ostream &operator<<(ostream &ost, const perps &C) {
  for (auto p : C)
    ost << "<" << p.first << ", " << p.second << ">"
        << "\t";
  return ost;
}

ostream &Game::operator<<(ostream &os, const Game::QP_Param &Q) {
  os << "Quadratic program with linear inequality constraints: " << '\n';
  os << Q.getNy() << " decision variables parametrized by " << Q.getNx()
     << " variables" << '\n';
  os << Q.getb().n_rows << " linear inequalities" << '\n' << '\n';
  return os;
}

void Game::MP_Param::write(string filename, bool) const {
  /**
   * @brief  Writes a given parameterized Mathematical program to a set of
   * files.
   *
   * Writes a given parameterized Mathematical program to a set of files.
   * One file is written for each attribute namely
   * 1. Game::MP_Param::Q
   * 2. Game::MP_Param::C
   * 3. Game::MP_Param::A
   * 4. Game::MP_Param::B
   * 5. Game::MP_Param::c
   * 6. Game::MP_Param::b
   *
   * To contrast see, Game::MP_Param::save where all details are written to a
   * single loadable file
   *
   */
  this->getQ().save(filename + "_Q.txt", arma::file_type::arma_ascii);
  this->getC().save(filename + "_C.txt", arma::file_type::arma_ascii);
  this->getA().save(filename + "_A.txt", arma::file_type::arma_ascii);
  this->getB().save(filename + "_B.txt", arma::file_type::arma_ascii);
  this->getc().save(filename + "_c.txt", arma::file_type::arma_ascii);
  this->getb().save(filename + "_b.txt", arma::file_type::arma_ascii);
}

void Game::QP_Param::write(string filename, bool append) const {
  ofstream file;
  file.open(filename, append ? ios::app : ios::out);
  file << *this;
  file << "\n\nOBJECTIVES\n";
  file << "Q:" << this->getQ();
  file << "C:" << this->getC();
  file << "c\n" << this->getc();
  file << "\n\nCONSTRAINTS\n";
  file << "A:" << this->getA();
  file << "B:" << this->getB();
  file << "b\n" << this->getb();
  file.close();
}

Game::MP_Param &Game::MP_Param::addDummy(unsigned int pars, unsigned int vars,
                                         int position)
/**
 * Adds dummy variables to a parameterized mathematical program
 * @p position dictates the position at which the parameters can be added. -1
 * for adding at the end.
 * @warning @p position cannot be set for @p vars. @p vars always added at the
 * end.
 */
{
  this->Nx += pars;
  this->Ny += vars;
  if (vars) {
    Q = resize_patch(Q, this->Ny, this->Ny);
    B = resize_patch(B, this->Ncons, this->Ny);
    c = resize_patch(c, this->Ny);
  }
  switch (position) {
  case -1:
    if (pars)
      A = resize_patch(A, this->Ncons, this->Nx);
    if (vars || pars)
      C = resize_patch(C, this->Ny, this->Nx);
    break;
  case 0:
    if (pars)
      A = arma::join_rows(arma::zeros<arma::sp_mat>(this->Ncons, pars), A);
    if (vars || pars) {
      C = resize_patch(C, this->Ny, C.n_cols);
      C = arma::join_rows(arma::zeros<arma::sp_mat>(this->Ny, pars), C);
    }
    break;
  default:
    if (pars) {
      arma::sp_mat A_temp =
          arma::join_rows(A.cols(0, position - 1),
                          arma::zeros<arma::sp_mat>(this->Ncons, pars));
      if (static_cast<unsigned int>(position) < A.n_cols) {
        A = arma::join_rows(A_temp, A.cols(position, A.n_cols - 1));
      } else {
        A = A_temp;
      }
    }
    if (vars || pars) {
      C = resize_patch(C, this->Ny, C.n_cols);
      arma::sp_mat C_temp = arma::join_rows(
          C.cols(0, position - 1), arma::zeros<arma::sp_mat>(this->Ny, pars));
      if (static_cast<unsigned int>(position) < C.n_cols) {
        C = arma::join_rows(C_temp, C.cols(position, C.n_cols - 1));
      } else {
        C = C_temp;
      }
    }
    break;
  };
  return *this;
}

unsigned int Game::MP_Param::size()
/** @brief Calculates @p Nx, @p Ny and @p Ncons
 *	Computes parameters in MP_Param:
 *		- Computes @p Ny as number of rows in MP_Param::Q
 * 		- Computes @p Nx as number of columns in MP_Param::C
 * 		- Computes @p Ncons as number of rows in MP_Param::b, i.e., the
 *RHS of the constraints
 *
 * 	For proper working, MP_Param::dataCheck() has to be run after this.
 * 	@returns @p Ny, Number of variables in the quadratic program, QP
 */
{
  this->Ny = this->Q.n_rows;
  this->Nx = this->C.n_cols;
  this->Ncons = this->b.size();
  return Ny;
}

Game::MP_Param &
Game::MP_Param::set(const arma::sp_mat &Q, const arma::sp_mat &C,
                    const arma::sp_mat &A, const arma::sp_mat &B,
                    const arma::vec &c, const arma::vec &b)
/// Setting the data, while keeping the input objects intact
{
  this->Q = (Q);
  this->C = (C);
  this->A = (A);
  this->B = (B);
  this->c = (c);
  this->b = (b);
  if (!finalize())
    throw string("Error in MP_Param::set: Invalid data");
  return *this;
}

Game::MP_Param &Game::MP_Param::set(arma::sp_mat &&Q, arma::sp_mat &&C,
                                    arma::sp_mat &&A, arma::sp_mat &&B,
                                    arma::vec &&c, arma::vec &&b)
/// Faster means to set data. But the input objects might be corrupted now.
{
  this->Q = move(Q);
  this->C = move(C);
  this->A = move(A);
  this->B = move(B);
  this->c = move(c);
  this->b = move(b);
  if (!finalize())
    throw string("Error in MP_Param::set: Invalid data");
  return *this;
}

Game::MP_Param &Game::MP_Param::set(const QP_objective &obj,
                                    const QP_constraints &cons) {
  return this->set(obj.Q, obj.C, cons.A, cons.B, obj.c, cons.b);
}

Game::MP_Param &Game::MP_Param::set(QP_objective &&obj, QP_constraints &&cons) {
  return this->set(obj.Q, obj.C, cons.A, cons.B, obj.c, cons.b);
}

bool Game::MP_Param::dataCheck(bool forcesymm) const
/** @brief Check that the data for the MP_Param class is valid
 * Always works after calls to MP_Param::size()
 * Checks that are done:
 * 		- Number of columns in @p Q is same as @p Ny (Q should be
 * square)
 * 		- Number of columns of @p A should be @p Nx
 * 		- Number of columns of @p B should be @p Ny
 * 		- Number of rows in @p C should be @p Ny
 * 		- Size of @p c should be @p Ny
 * 		- @p A and @p B should have the same number of rows, equal to @p
 * Ncons
 * 		- if @p forcesymm is @p true, then Q should be symmetric
 *
 * 	@returns true if all above checks are cleared. false otherwise.
 */
{
  if (forcesymm) {
  }
  if (this->Q.n_cols != Ny) {
    return false;
  }
  if (this->A.n_cols != Nx) {
    return false;
  } // Rest are matrix size compatibility checks
  if (this->B.n_cols != Ny) {
    return false;
  }
  if (this->C.n_rows != Ny) {
    return false;
  }
  if (this->c.size() != Ny) {
    return false;
  }
  if (this->A.n_rows != Ncons) {
    return false;
  }
  if (this->B.n_rows != Ncons) {
    return false;
  }
  return true;
}

bool Game::MP_Param::dataCheck(const QP_objective &obj,
                               const QP_constraints &cons, bool checkobj,
                               bool checkcons) {
  unsigned int Ny = obj.Q.n_rows;
  unsigned int Nx = obj.C.n_cols;
  unsigned int Ncons = cons.b.size();
  if (checkobj && obj.Q.n_cols != Ny) {
    return false;
  }
  if (checkobj && obj.C.n_rows != Ny) {
    return false;
  }
  if (checkobj && obj.c.size() != Ny) {
    return false;
  }
  if (checkcons && cons.A.n_cols != Nx) {
    return false;
  } // Rest are matrix size compatibility checks
  if (checkcons && cons.B.n_cols != Ny) {
    return false;
  }
  if (checkcons && cons.A.n_rows != Ncons) {
    return false;
  }
  if (checkcons && cons.B.n_rows != Ncons) {
    return false;
  }
  return true;
}

bool Game::QP_Param::operator==(const QP_Param &Q2) const {
  if (!Game::isZero(this->Q - Q2.getQ()))
    return false;
  if (!Game::isZero(this->C - Q2.getC()))
    return false;
  if (!Game::isZero(this->A - Q2.getA()))
    return false;
  if (!Game::isZero(this->B - Q2.getB()))
    return false;
  if (!Game::isZero(this->c - Q2.getc()))
    return false;
  if (!Game::isZero(this->b - Q2.getb()))
    return false;
  return true;
}

int Game::QP_Param::make_yQy()
/// Adds the Gurobi Quadratic objective to the Gurobi model @p QuadModel.
{
  if (this->made_yQy)
    return 0;
  GRBVar y[this->Ny];
  for (unsigned int i = 0; i < Ny; i++)
    y[i] = this->QuadModel.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS,
                                  "y_" + to_string(i));
  GRBQuadExpr yQy{0};
  for (auto val = Q.begin(); val != Q.end(); ++val) {
    unsigned int i, j;
    double value = (*val);
    i = val.row();
    j = val.col();
    yQy += 0.5 * y[i] * value * y[j];
  }
  QuadModel.setObjective(yQy, GRB_MINIMIZE);
  QuadModel.update();
  this->made_yQy = true;
  return 0;
}

unique_ptr<GRBModel> Game::QP_Param::solveFixed(
    arma::vec x ///< Other players' decisions
    )           /**
                 * Given a value for the parameters @f$x@f$ in the definition of QP_Param,
                 * solve           the parameterized quadratic program to  optimality.
                 *
                 * In terms of game theory, this can be viewed as <i>the best response</i>
                 * for a           set of decisions by other players.
                 *
                 */
{
  this->make_yQy(); /// @throws GRBException if argument vector size is not
  /// compatible with the Game::QP_Param definition.
  if (x.size() != this->Nx)
    throw "Game::QP_Param::solveFixed: Invalid argument size: " +
        to_string(x.size()) + " != " + to_string(Nx);
  unique_ptr<GRBModel> model(new GRBModel(this->QuadModel));
  try {
    GRBQuadExpr yQy = model->getObjective();
    arma::vec Cx, Ax;
    Cx = this->C * x;
    Ax = this->A * x;
    GRBVar y[this->Ny];
    for (unsigned int i = 0; i < this->Ny; i++) {
      y[i] = model->getVarByName("y_" + to_string(i));
      yQy += (Cx[i] + c[i]) * y[i];
    }
    model->setObjective(yQy, GRB_MINIMIZE);
    for (unsigned int i = 0; i < this->Ncons; i++) {
      GRBLinExpr LHS{0};
      for (auto j = B.begin_row(i); j != B.end_row(i); ++j)
        LHS += (*j) * y[j.col()];
      model->addConstr(LHS, GRB_LESS_EQUAL, b[i] - Ax[i]);
    }
    model->update();
    model->set(GRB_IntParam_OutputFlag, 0);
    model->optimize();
  } catch (const char *e) {
    cerr << " Error in Game::QP_Param::solveFixed: " << e << '\n';
    throw;
  } catch (string e) {
    cerr << "String: Error in Game::QP_Param::solveFixed: " << e << '\n';
    throw;
  } catch (exception &e) {
    cerr << "Exception: Error in Game::QP_Param::solveFixed: " << e.what()
         << '\n';
    throw;
  } catch (GRBException &e) {
    cerr << "GRBException: Error in Game::QP_Param::solveFixed: "
         << e.getErrorCode() << "; " << e.getMessage() << '\n';
    throw;
  }
  return model;
}

Game::QP_Param &Game::QP_Param::addDummy(unsigned int pars, unsigned int vars,
                                         int position)
/**
 * @warning You might have to rerun QP_Param::KKT since you have now changed the
 * QP.
 * @warning This implies you might have to rerun NashGame::FormulateLCP again
 * too.
 */
{
  // if ((pars || vars))
  // BOOST_LOG_TRIVIAL(trace)
  // << "From Game::QP_Param::addDummyVars:\t You might have to rerun
  // Games::QP_Param::KKT since you have now changed the number of variables in
  // the NashGame.";

  // Call the superclass function
  try {
    MP_Param::addDummy(pars, vars, position);
  } catch (const char *e) {
    cerr << " Error in Game::QP_Param::addDummy: " << e << '\n';
    throw;
  } catch (string e) {
    cerr << "String: Error in Game::QP_Param::addDummy: " << e << '\n';
    throw;
  } catch (exception &e) {
    cerr << "Exception: Error in Game::QP_Param::addDummy: " << e.what()
         << '\n';
    throw;
  }
  return *this;
}
unsigned int Game::QP_Param::KKT(arma::sp_mat &M, arma::sp_mat &N,
                                 arma::vec &q) const
/// @brief Compute the KKT conditions for the given QP
/**
 * Writes the KKT condition of the parameterized QP
 * As per the convention, y is the decision variable for the QP and
 * that is parameterized in x
 * The KKT conditions are
 * \f$0 \leq y \perp  My + Nx + q \geq 0\f$
 */
{
  if (!this->dataCheck()) {
    throw string("Inconsistent data for KKT of Game::QP_Param::KKT");
    return 0;
  }
  M = arma::join_cols( // In armadillo join_cols(A, B) is same as [A;B] in
                       // Matlab
                       //  join_rows(A, B) is same as [A B] in Matlab
      arma::join_rows(this->Q, this->B.t()),
      arma::join_rows(-this->B,
                      arma::zeros<arma::sp_mat>(this->Ncons, this->Ncons)));
  // M.print_dense();
  N = arma::join_cols(this->C, -this->A);
  // N.print_dense();
  q = arma::join_cols(this->c, this->b);
  // q.print();
  return M.n_rows;
}

Game::QP_Param &
Game::QP_Param::set(const arma::sp_mat &Q, const arma::sp_mat &C,
                    const arma::sp_mat &A, const arma::sp_mat &B,
                    const arma::vec &c, const arma::vec &b)
/// Setting the data, while keeping the input objects intact
{
  this->made_yQy = false;
  try {
    MP_Param::set(Q, C, A, B, c, b);
  } catch (string &e) {
    cerr << "String: " << e << '\n';
    throw string("Error in QP_Param::set: Invalid Data");
  }
  return *this;
}

Game::QP_Param &Game::QP_Param::set(arma::sp_mat &&Q, arma::sp_mat &&C,
                                    arma::sp_mat &&A, arma::sp_mat &&B,
                                    arma::vec &&c, arma::vec &&b)
/// Faster means to set data. But the input objects might be corrupted now.
{
  this->made_yQy = false;
  try {
    MP_Param::set(Q, C, A, B, c, b);
  } catch (string &e) {
    cerr << "String: " << e << '\n';
    throw string("Error in QP_Param::set: Invalid Data");
  }
  return *this;
}

Game::QP_Param &Game::QP_Param::set(QP_objective &&obj, QP_constraints &&cons)
/// Setting the data with the inputs being a struct Game::QP_objective and
/// struct Game::QP_constraints
{
  return this->set(move(obj.Q), move(obj.C), move(cons.A), move(cons.B),
                   move(obj.c), move(cons.b));
}

Game::QP_Param &Game::QP_Param::set(const QP_objective &obj,
                                    const QP_constraints &cons) {
  return this->set(obj.Q, obj.C, cons.A, cons.B, obj.c, cons.b);
}

double Game::QP_Param::computeObjective(const arma::vec &y, const arma::vec &x,
                                        bool checkFeas, double tol) const {
  /**
   * Computes @f$\frac{1}{2} y^TQy + (Cx)^Ty @f$ given the input values @p y and
   * @p x.
   * @param checkFeas if @p true, checks if the given @f$(x,y)@f$ satisfies the
   * constraints of the problem, namely @f$Ax + By \leq b@f$.
   */
  if (y.n_rows != this->getNy())
    throw string("Error in QP_Param::computeObjective: Invalid size of y");
  if (x.n_rows != this->getNx())
    throw string("Error in QP_Param::computeObjective: Invalid size of x");
  if (checkFeas) {
    arma::vec slack = A * x + B * y - b;
    if (slack.n_rows) // if infeasible
      if (slack.max() >= tol)
        return GRB_INFINITY;
    if (y.min() <= -tol) // if infeasible
      return GRB_INFINITY;
  }
  arma::vec obj = 0.5 * y.t() * Q * y + (C * x).t() * y + c.t() * y;
  return obj(0);
}

void Game::QP_Param::save(string filename, bool erase) const {
  /**
   * The Game::QP_Param object hence stored can be loaded back using
   * Game::QP_Param::load
   */
  Utils::appendSave(string("QP_Param"), filename, erase);
  Utils::appendSave(this->Q, filename, string("QP_Param::Q"), false);
  Utils::appendSave(this->A, filename, string("QP_Param::A"), false);
  Utils::appendSave(this->B, filename, string("QP_Param::B"), false);
  Utils::appendSave(this->C, filename, string("QP_Param::C"), false);
  Utils::appendSave(this->b, filename, string("QP_Param::b"), false);
  Utils::appendSave(this->c, filename, string("QP_Param::c"), false);
  BOOST_LOG_TRIVIAL(trace) << "Saved QP_Param to file " << filename;
}

long int Game::QP_Param::load(string filename, long int pos) {
  /**
   * @details  Before calling this function, use the constructor
   * QP_Param::QP_Param(GRBEnv *env) to initialize.
   *
   * Example usage:
   * @code{.cpp}
   * int main()
   * {
   * 		GRBEnv env;
   * 		Game::QP_Param q1(&env);
   * 		q1.load("./dat/q1data.dat");
   * 		std::cout<<q1<<'\n';
   * 		return 0;
   * }
   * @endcode
   *
   */
  arma::sp_mat Q, A, B, C;
  arma::vec c, b;
  string headercheck;
  pos = Utils::appendRead(headercheck, filename, pos);
  if (headercheck != "QP_Param")
    throw string("Error in QP_Param::load: In valid header - ") + headercheck;
  pos = Utils::appendRead(Q, filename, pos, string("QP_Param::Q"));
  pos = Utils::appendRead(A, filename, pos, string("QP_Param::A"));
  pos = Utils::appendRead(B, filename, pos, string("QP_Param::B"));
  pos = Utils::appendRead(C, filename, pos, string("QP_Param::C"));
  pos = Utils::appendRead(b, filename, pos, string("QP_Param::b"));
  pos = Utils::appendRead(c, filename, pos, string("QP_Param::c"));
  this->set(Q, C, A, B, c, b);
  return pos;
}
Game::NashGame::NashGame(GRBEnv *e, vector<shared_ptr<QP_Param>> Players,
                         arma::sp_mat MC, arma::vec MCRHS,
                         unsigned int n_LeadVar, arma::sp_mat LeadA,
                         arma::vec LeadRHS)
    : env{e}, LeaderConstraints{LeadA}, LeaderConsRHS{LeadRHS}
/**
 * @brief
 * Construct a NashGame by giving a vector of pointers to
 * QP_Param, defining each player's game
 * A set of Market clearing constraints and its RHS
 * And if there are leader variables, the number of leader vars.
 * @details
 * Have a vector of pointers to Game::QP_Param ready such that
 * the variables are separated in \f$x^{i}\f$ and \f$x^{-i}\f$
 * format.
 *
 * In the correct ordering of variables, have the
 * Market clearing equations ready.
 *
 * Now call this constructor.
 * It will allocate appropriate space for the dual variables
 * for each player.
 *
 */
{
  // Setting the class variables
  this->n_LeadVar = n_LeadVar;
  this->Nplayers = Players.size();
  this->Players = Players;
  this->MarketClearing = MC;
  this->MCRHS = MCRHS;
  // Setting the size of class variable vectors
  this->primal_position.resize(this->Nplayers + 1);
  this->dual_position.resize(this->Nplayers + 1);
  this->set_positions();
}

Game::NashGame::NashGame(const NashGame &N)
    : env{N.env}, LeaderConstraints{N.LeaderConstraints},
      LeaderConsRHS{N.LeaderConsRHS}, Nplayers{N.Nplayers}, Players{N.Players},
      MarketClearing{N.MarketClearing}, MCRHS{N.MCRHS}, n_LeadVar{N.n_LeadVar} {
  // Setting the size of class variable vectors
  this->primal_position.resize(this->Nplayers + 1);
  this->dual_position.resize(this->Nplayers + 1);
  this->set_positions();
}

void Game::NashGame::save(string filename, bool erase) const {
  Utils::appendSave(string("NashGame"), filename, erase);
  Utils::appendSave(this->Nplayers, filename, string("NashGame::Nplayers"),
                    false);
  for (unsigned int i = 0; i < this->Nplayers; ++i)
    this->Players.at(i)->save(filename, false);
  Utils::appendSave(this->MarketClearing, filename,
                    string("NashGame::MarketClearing"), false);
  Utils::appendSave(this->MCRHS, filename, string("NashGame::MCRHS"), false);
  Utils::appendSave(this->LeaderConstraints, filename,
                    string("NashGame::LeaderConstraints"), false);
  Utils::appendSave(this->LeaderConsRHS, filename,
                    string("NashGame::LeaderConsRHS"), false);
  Utils::appendSave(this->n_LeadVar, filename, string("NashGame::n_LeadVar"),
                    false);
  BOOST_LOG_TRIVIAL(trace) << "Saved NashGame to file " << filename;
}

long int Game::NashGame::load(string filename, long int pos) {
  /**
   * @brief Loads the @p NashGame object stored in a file.  Before calling this
   * function, use the constructor NashGame::NashGame(GRBEnv *env) to
   * initialize.
   * @details Loads the @p NashGame object stored in a file.  Before calling
   * this function, use the constructor NashGame::NashGame(GRBEnv *env) to
   * initialize. Example usage:
   * @code{.cpp}
   * int main()
   * {
   * 		GRBEnv env;
   * 		Game::NashGame N(&env);
   * 		N.load("./dat/Ndata.dat");
   * 		std::cout<<N<<'\n';
   * 		return 0;
   * }
   * @endcode
   *
   */
  if (!this->env)
    throw string("Error in NashGame::load: To load NashGame from file, it has "
                 "to be constructed using NashGame(GRBEnv*) constructor");
  string headercheck;
  pos = Utils::appendRead(headercheck, filename, pos);
  if (headercheck != "NashGame")
    throw string("Error in NashGame::load: In valid header - ") + headercheck;
  unsigned int Nplayers;
  pos =
      Utils::appendRead(Nplayers, filename, pos, string("NashGame::Nplayers"));
  vector<shared_ptr<QP_Param>> Players;
  Players.resize(Nplayers);
  for (unsigned int i = 0; i < Nplayers; ++i) {
    // Players.at(i) = std::make_shared<Game::QP_Param>(this->env);
    auto temp = shared_ptr<Game::QP_Param>(new Game::QP_Param(this->env));
    Players.at(i) = temp;
    pos = Players.at(i)->load(filename, pos);
  }
  arma::sp_mat MarketClearing;
  pos = Utils::appendRead(MarketClearing, filename, pos,
                          string("NashGame::MarketClearing"));
  arma::vec MCRHS;
  pos = Utils::appendRead(MCRHS, filename, pos, string("NashGame::MCRHS"));
  arma::sp_mat LeaderConstraints;
  pos = Utils::appendRead(LeaderConstraints, filename, pos,
                          string("NashGame::LeaderConstraints"));
  arma::vec LeaderConsRHS;
  pos = Utils::appendRead(LeaderConsRHS, filename, pos,
                          string("NashGame::LeaderConsRHS"));
  unsigned int n_LeadVar;
  pos = Utils::appendRead(n_LeadVar, filename, pos,
                          string("NashGame::n_LeadVar"));
  // Setting the class variables
  this->n_LeadVar = n_LeadVar;
  this->Players = Players;
  this->Nplayers = Nplayers;
  this->MarketClearing = MarketClearing;
  this->MCRHS = MCRHS;
  // Setting the size of class variable vectors
  this->primal_position.resize(this->Nplayers + 1);
  this->dual_position.resize(this->Nplayers + 1);
  this->set_positions();
  return pos;
}

void Game::NashGame::set_positions()
/**
 * Stores the position of each players' primal and dual variables. Also
 allocates Leader's position appropriately.
 * The ordering is according to the columns of
         @image html FormulateLCP.png
         @image latex FormulateLCP.png
 */
{
  // Defining the variable value
  unsigned int pr_cnt{0},
      dl_cnt{0}; // Temporary variables - primal count and dual count
  for (unsigned int i = 0; i < Nplayers; i++) {
    primal_position.at(i) = pr_cnt;
    pr_cnt += Players.at(i)->getNy();
  }

  // Pushing back the end of primal position
  primal_position.at(Nplayers) = (pr_cnt);
  dl_cnt = pr_cnt; // From now on, the space is for dual variables.
  this->MC_dual_position = dl_cnt;
  this->Leader_position = dl_cnt + MCRHS.n_rows;
  dl_cnt += (MCRHS.n_rows + n_LeadVar);
  for (unsigned int i = 0; i < Nplayers; i++) {
    dual_position.at(i) = dl_cnt;
    dl_cnt += Players.at(i)->getb().n_rows;
  }
  // Pushing back the end of dual position
  dual_position.at(Nplayers) = (dl_cnt);
}

const Game::NashGame &Game::NashGame::FormulateLCP(
    arma::sp_mat &M, ///< Where the output  M is stored and returned.
    arma::vec &q,    ///< Where the output  q is stored and returned.
    perps &Compl, ///< Says which equations are complementary to which variables
    bool writeToFile, ///< If  true, writes  M and  q to file.k
    string M_name,    ///< File name to be used to write  M
    string q_name     ///< File name to be used to write  M
    ) const {
  /// @brief Formulates the LCP corresponding to the Nash game.
  /// @warning Does not return the leader constraints. Use
  /// NashGame::RewriteLeadCons() to handle them
  /**
 * Computes the KKT conditions for each Player, calling QP_Param::KKT.
 Arranges them systematically to return M, q
 * as an LCP @f$0\leq q \perp Mx+q \geq 0 @f$.
         The way the variables of the players get distributed is shown in the
 image below
         @image html FormulateLCP.png
         @image latex FormulateLCP.png
 */

  // To store the individual KKT conditions for each player.
  vector<arma::sp_mat> Mi(Nplayers), Ni(Nplayers);
  vector<arma::vec> qi(Nplayers);

  unsigned int NvarFollow{0}, NvarLead{0};
  NvarLead =
      this->dual_position.back(); // Number of Leader variables (all variables)
  NvarFollow = NvarLead - this->n_LeadVar;
  M.zeros(NvarFollow, NvarLead);
  q.zeros(NvarFollow);
  // Get the KKT conditions for each player

  for (unsigned int i = 0; i < Nplayers; i++) {
    this->Players[i]->KKT(Mi[i], Ni[i], qi[i]);
    unsigned int Nprim, Ndual;
    Nprim = this->Players[i]->getNy();
    Ndual = this->Players[i]->getA().n_rows;
    // Adding the primal equations
    // Region 1 in Formulate LCP.ipe
    BOOST_LOG_TRIVIAL(trace) << "Game::NashGame::FormulateLCP: Region 1";
    if (i > 0) { // For the first player, no need to add anything 'before' 0-th
      // position
      M.submat(this->primal_position.at(i), 0,
               this->primal_position.at(i + 1) - 1,
               this->primal_position.at(i) - 1) =
          Ni[i].submat(0, 0, Nprim - 1, this->primal_position.at(i) - 1);
    }
    // Region 2 in Formulate LCP.ipe
    BOOST_LOG_TRIVIAL(trace) << "Game::NashGame::FormulateLCP: Region 2";
    M.submat(this->primal_position.at(i), this->primal_position.at(i),
             this->primal_position.at(i + 1) - 1,
             this->primal_position.at(i + 1) - 1) =
        Mi[i].submat(0, 0, Nprim - 1, Nprim - 1);
    // Region 3 in Formulate LCP.ipe
    BOOST_LOG_TRIVIAL(trace) << "Game::NashGame::FormulateLCP: Region 3";
    if (this->primal_position.at(i + 1) != this->dual_position.at(0)) {
      M.submat(this->primal_position.at(i), this->primal_position.at(i + 1),
               this->primal_position.at(i + 1) - 1,
               this->dual_position.at(0) - 1) =
          Ni[i].submat(0, this->primal_position.at(i), Nprim - 1,
                       Ni[i].n_cols - 1);
    }
    // Region 4 in Formulate LCP.ipe
    BOOST_LOG_TRIVIAL(trace) << "Game::NashGame::FormulateLCP: Region 4";
    if (this->dual_position.at(i) != this->dual_position.at(i + 1)) {
      M.submat(this->primal_position.at(i), this->dual_position.at(i),
               this->primal_position.at(i + 1) - 1,
               this->dual_position.at(i + 1) - 1) =
          Mi[i].submat(0, Nprim, Nprim - 1, Nprim + Ndual - 1);
    }
    // RHS
    BOOST_LOG_TRIVIAL(trace) << "Game::NashGame::FormulateLCP: Region RHS";
    q.subvec(this->primal_position.at(i), this->primal_position.at(i + 1) - 1) =
        qi[i].subvec(0, Nprim - 1);
    for (unsigned int j = this->primal_position.at(i);
         j < this->primal_position.at(i + 1); j++)
      Compl.push_back({j, j});
    // Adding the dual equations
    // Region 5 in Formulate LCP.ipe
    BOOST_LOG_TRIVIAL(trace) << "Game::NashGame::FormulateLCP: Region 5";
    if (Ndual > 0) {
      if (i > 0) // For the first player, no need to add anything 'before' 0-th
        // position
        M.submat(this->dual_position.at(i) - n_LeadVar, 0,
                 this->dual_position.at(i + 1) - n_LeadVar - 1,
                 this->primal_position.at(i) - 1) =
            Ni[i].submat(Nprim, 0, Ni[i].n_rows - 1,
                         this->primal_position.at(i) - 1);
      // Region 6 in Formulate LCP.ipe
      BOOST_LOG_TRIVIAL(trace) << "Game::NashGame::FormulateLCP: Region 6";
      M.submat(this->dual_position.at(i) - n_LeadVar,
               this->primal_position.at(i),
               this->dual_position.at(i + 1) - n_LeadVar - 1,
               this->primal_position.at(i + 1) - 1) =
          Mi[i].submat(Nprim, 0, Nprim + Ndual - 1, Nprim - 1);
      // Region 7 in Formulate LCP.ipe
      BOOST_LOG_TRIVIAL(trace) << "Game::NashGame::FormulateLCP: Region 7";
      if (this->dual_position.at(0) != this->primal_position.at(i + 1)) {
        M.submat(this->dual_position.at(i) - n_LeadVar,
                 this->primal_position.at(i + 1),
                 this->dual_position.at(i + 1) - n_LeadVar - 1,
                 this->dual_position.at(0) - 1) =
            Ni[i].submat(Nprim, this->primal_position.at(i), Ni[i].n_rows - 1,
                         Ni[i].n_cols - 1);
      }
      // Region 8 in Formulate LCP.ipe
      BOOST_LOG_TRIVIAL(trace) << "Game::NashGame::FormulateLCP: Region 8";
      M.submat(this->dual_position.at(i) - n_LeadVar, this->dual_position.at(i),
               this->dual_position.at(i + 1) - n_LeadVar - 1,
               this->dual_position.at(i + 1) - 1) =
          Mi[i].submat(Nprim, Nprim, Nprim + Ndual - 1, Nprim + Ndual - 1);
      // RHS
      BOOST_LOG_TRIVIAL(trace) << "Game::NashGame::FormulateLCP: Region RHS";
      q.subvec(this->dual_position.at(i) - n_LeadVar,
               this->dual_position.at(i + 1) - n_LeadVar - 1) =
          qi[i].subvec(Nprim, qi[i].n_rows - 1);
      for (unsigned int j = this->dual_position.at(i) - n_LeadVar;
           j < this->dual_position.at(i + 1) - n_LeadVar; j++)
        Compl.push_back({j, j + n_LeadVar});
    }
  }
  BOOST_LOG_TRIVIAL(trace) << "Game::NashGame::FormulateLCP: MC RHS";
  if (this->MCRHS.n_elem >= 1) // It is possible that it is a Cournot game and
                               // there are no MC conditions!
  {
    M.submat(this->MC_dual_position, 0, this->Leader_position - 1,
             this->dual_position.at(0) - 1) = this->MarketClearing;
    q.subvec(this->MC_dual_position, this->Leader_position - 1) = -this->MCRHS;
    for (unsigned int j = this->MC_dual_position; j < this->Leader_position;
         j++)
      Compl.push_back({j, j});
  }
  if (writeToFile) {
    M.save(M_name, arma::coord_ascii);
    q.save(q_name, arma::arma_ascii);
  }
  return *this;
}

arma::sp_mat Game::NashGame::RewriteLeadCons() const
/** @brief Rewrites leader constraint adjusting for dual variables.
 * Rewrites leader constraints given earlier with added empty columns and spaces
 * corresponding to Market clearing duals and other equation duals.
 *
 * This becomes important if the Lower level complementarity problem is passed
 * to LCP with upper level constraints.
 */
{
  arma::sp_mat A_in = this->LeaderConstraints;
  arma::sp_mat A_out_expl, A_out_MC, A_out;
  unsigned int NvarLead{0};
  NvarLead =
      this->dual_position.back(); // Number of Leader variables (all variables)
  // NvarFollow = NvarLead - this->n_LeadVar;

  unsigned int n_Row, n_Col;
  n_Row = A_in.n_rows;
  n_Col = A_in.n_cols;
  A_out_expl.zeros(n_Row, NvarLead);
  A_out_MC.zeros(2 * this->MarketClearing.n_rows, NvarLead);

  try {
    if (A_in.n_rows) {
      // Primal variables i.e., everything before MCduals are the same!
      A_out_expl.cols(0, this->MC_dual_position - 1) =
          A_in.cols(0, this->MC_dual_position - 1);
      A_out_expl.cols(this->Leader_position, this->dual_position.at(0) - 1) =
          A_in.cols(this->MC_dual_position, n_Col - 1);
    }
    if (this->MCRHS.n_rows) {
      // MC constraints can be written as if they are leader constraints
      A_out_MC.submat(0, 0, this->MCRHS.n_rows - 1,
                      this->dual_position.at(0) - 1) = this->MarketClearing;
      A_out_MC.submat(this->MCRHS.n_rows, 0, 2 * this->MCRHS.n_rows - 1,
                      this->dual_position.at(0) - 1) = -this->MarketClearing;
    }
    return arma::join_cols(A_out_expl, A_out_MC);
  } catch (const char *e) {
    cerr << "Error in NashGame::RewriteLeadCons: " << e << '\n';
    throw;
  } catch (string e) {
    cerr << "String: Error in NashGame::RewriteLeadCons: " << e << '\n';
    throw;
  } catch (exception &e) {
    cerr << "Exception: Error in NashGame::RewriteLeadCons: " << e.what()
         << '\n';
    throw;
  }
}

Game::NashGame &Game::NashGame::addDummy(unsigned int par, int position)
/**
 * @brief Add dummy variables in a NashGame object.
 * @details Add extra variables at the end of the problem. These are just zero
 * columns that don't feature in the problem anywhere. They are of importance
 * only where the NashGame gets converted into an LCP and gets parametrized.
 * Typically, they appear in the upper level objective in such a case.
 */
{
  for (auto &q : this->Players)
    q->addDummy(par, 0, position);

  this->n_LeadVar += par;
  if (this->LeaderConstraints.n_rows) {
    auto nnR = this->LeaderConstraints.n_rows;
    auto nnC = this->LeaderConstraints.n_cols;
    switch (position) {
    case -1:
      this->LeaderConstraints =
          resize_patch(this->LeaderConstraints, nnR, nnC + par);
      break;
    case 0:
      this->LeaderConstraints = arma::join_rows(
          arma::zeros<arma::sp_mat>(nnR, par), this->LeaderConstraints);
      break;
    default:
      arma::sp_mat lC = arma::join_rows(LeaderConstraints.cols(0, position - 1),
                                        arma::zeros<arma::sp_mat>(nnR, par));

      this->LeaderConstraints =
          arma::join_rows(lC, LeaderConstraints.cols(position, nnC - 1));
      break;
    };
  }
  this->set_positions();
  return *this;
}

Game::NashGame &Game::NashGame::addLeadCons(const arma::vec &a, double b)
/**
 * @brief Adds Leader constraint to a NashGame object.
 * @details In case common constraint to all followers is to be added (like  a
 * leader constraint in an MPEC), this function can be used. It adds a single
 * constraint @f$ a^Tx \leq b@f$
 */
{
  auto nC = this->LeaderConstraints.n_cols;
  if (a.n_elem != nC)
    throw string("Error in NashGame::addLeadCons: Leader constraint size "
                 "incompatible --- ") +
        to_string(a.n_elem) + string(" != ") + to_string(nC);
  auto nR = this->LeaderConstraints.n_rows;
  this->LeaderConstraints = resize_patch(this->LeaderConstraints, nR + 1, nC);
  // (static_cast<arma::mat>(a)).t();	// Apparently this is not reqd! a.t()
  // already works in newer versions of armadillo
  LeaderConstraints.row(nR) = a.t();
  this->LeaderConsRHS = resize_patch(this->LeaderConsRHS, nR + 1);
  this->LeaderConsRHS(nR) = b;
  return *this;
}

void Game::NashGame::write(string filename, bool append, bool KKT) const {
  ofstream file;
  file.open(filename + ".nash", append ? ios::app : ios::out);
  file << *this;
  file << "\n\n\n\n\n\n\n";
  file << "\nLeaderConstraints: " << this->LeaderConstraints;
  file << "\nLeaderConsRHS\n" << this->LeaderConsRHS;
  file << "\nMarketClearing: " << this->MarketClearing;
  file << "\nMCRHS\n" << this->MCRHS;

  file.close();

  // this->LeaderConstraints.save(filename+"_LeaderConstraints.txt",
  // arma::file_type::arma_ascii);
  // this->LeaderConsRHS.save(filename+"_LeaderConsRHS.txt",
  // arma::file_type::arma_ascii);
  // this->MarketClearing.save(filename+"_MarketClearing.txt",
  // arma::file_type::arma_ascii); this->MCRHS.save(filename+"_MCRHS.txt",
  // arma::file_type::arma_ascii);

  int count{0};
  for (const auto &pl : this->Players) {
    // pl->QP_Param::write(filename+"_Players_"+to_string(count++), append);
    file << "--------------------------------------------------\n";
    file.open(filename + ".nash", ios::app);
    file << "\n\n\n\n PLAYER " << count++ << "\n\n";
    file.close();
    pl->QP_Param::write(filename + ".nash", true);
  }

  file.open(filename + ".nash", ios::app);
  file << "--------------------------------------------------\n";
  file << "\nPrimal Positions:\t";
  for (const auto pos : primal_position)
    file << pos << "  ";
  file << "\nDual Positions:\t";
  for (const auto pos : dual_position)
    file << pos << "  ";
  file << "\nMC dual position:\t" << this->MC_dual_position;
  file << "\nLeader position:\t" << this->Leader_position;
  file << "\nnLeader:\t" << this->n_LeadVar;

  if (KKT) {
    arma::sp_mat M;
    arma::vec q;
    perps Compl;
    this->FormulateLCP(M, q, Compl);
    file << "\n\n\n KKT CONDITIONS - LCP\n";
    file << "\nM: " << M;
    file << "\nq:\n" << q;
    file << "\n Complementarities:\n";
    for (const auto &p : Compl)
      file << "<" << p.first << ", " << p.second << ">"
           << "\t";
  }

  file << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";

  file.close();
}

unique_ptr<GRBModel> Game::NashGame::Respond(
    unsigned int player, ///< Player whose optimal response is to be computed
    const arma::vec &x,  ///< A vector of pure strategies (either for all
    ///< players or all other players)
    bool fullvec ///< Is @p x strategy of all players? (including player @p
                 ///< player)
    ) const
/**
 * @brief Given the decision of other players, find the optimal response for
 * player in position @p player
 * @details
 * Given the strategy of each player, returns a Gurobi Model that has the
 * optimal strategy of the player at position @p player.
 * @returns A unique_ptr to GRBModel
 *
 */
{
  arma::vec solOther;
  unsigned int nVar{this->getNprimals() + this->getNshadow() +
                    this->getNleaderVars()};
  unsigned int nStart, nEnd;
  nStart = this->primal_position.at(
      player); // Start of the player-th player's primals
  nEnd = this->primal_position.at(
      player + 1); // Start of the player+1-th player's primals or LeaderVrs if
  // player is the last player.
  if (fullvec) {
    solOther.zeros(nVar - nEnd + nStart);
    if (nStart > 0)
      solOther.subvec(0, nStart - 1) = x.subvec(0, nStart - 1);
    if (nEnd < nVar)
      solOther.subvec(nStart, nVar + nStart - nEnd - 1) =
          x.subvec(nEnd,
                   nVar - 1); // Discard any dual variables in x
  } else {
    solOther.zeros(nVar - nEnd + nStart);
    solOther = x.subvec(0, nVar - nEnd + nStart -
                               1); // Discard any dual variables in x
  }

  return this->Players.at(player)->solveFixed(solOther);
}

double Game::NashGame::RespondSol(
    arma::vec &sol,      ///< [out] Optimal response
    unsigned int player, ///< Player whose optimal response is to be computed
    const arma::vec &x,  ///< A vector of pure strategies (either for all
    ///< players or all other players)
    bool fullvec ///< Is @p x strategy of all players? (including player @p
                 ///< player)
    ) const {
  /**
   * @brief Returns the optimal objective value that is obtainable for the
   * player @p player given the decision @p x of all other players.
   * @details
   * Calls Game::NashGame::Respond and obtains the unique_ptr to GRBModel of
   * best response by player @p player. Then solves the model and returns the
   * appropriate objective value.
   * @returns The optimal objective value for the player @p player.
   */
  auto model = this->Respond(player, x, fullvec);
  // Check if the model is solved optimally
  const int status = model->get(GRB_IntAttr_Status);
  if (status == GRB_OPTIMAL) {
    unsigned int Nx =
        this->primal_position.at(player + 1) - this->primal_position.at(player);
    sol.zeros(Nx);
    for (unsigned int i = 0; i < Nx; ++i)
      sol.at(i) =
          model->getVarByName("y_" + to_string(i)).get(GRB_DoubleAttr_X);

    return model->get(GRB_DoubleAttr_ObjVal);
  } else
    return GRB_INFINITY;
}

arma::vec Game::NashGame::ComputeQPObjvals(const arma::vec &x,
                                           bool checkFeas) const {
  /**
   * @brief Computes players' objective
   * @details
   * Computes the objective value of <i> each </i> player in the Game::NashGame
   * object.
   * @returns An arma::vec with the objective values.
   */
  arma::vec vals;
  vals.zeros(this->Nplayers);
  for (unsigned int i = 0; i < this->Nplayers; ++i) {
    unsigned int nVar{this->getNprimals() + this->getNshadow() +
                      this->getNleaderVars()};
    unsigned int nStart, nEnd;
    nStart = this->primal_position.at(i);
    nEnd = this->primal_position.at(i + 1);

    arma::vec x_i, x_minus_i;

    x_minus_i.zeros(nVar - nEnd + nStart);
    if (nStart > 0) {
      x_minus_i.subvec(0, nStart - 1) = x.subvec(0, nStart - 1);
    }
    if (nEnd < nVar) {
      x_minus_i.subvec(nStart, nVar + nStart - nEnd - 1) =
          x.subvec(nEnd,
                   nVar - 1); // Discard any dual variables in x
    }

    x_i = x.subvec(nStart, nEnd - 1);

    vals.at(i) =
        this->Players.at(i)->computeObjective(x_i, x_minus_i, checkFeas);
  }

  return vals;
}

bool Game::NashGame::isSolved(const arma::vec &sol, unsigned int &violPlayer,
                              arma::vec &violSol, double tol) const {
  /**
   * @brief Checks if the Nash game is solved.
   * @details
   * Checks if the Nash game is solved, if not provides a proof of deviation
   * @param[in] sol - The vector of pure strategies for the Nash Game
   * @param[out] violPlayer - Index of the player with profitable deviation
   * @param[out] violSol - The pure strategy for that player - which gives a
   * profitable deviation
   * @param[in] tol - If the additional profit is smaller than this, then it is
   * not considered a profitable deviation.
   */
  arma::vec objvals = this->ComputeQPObjvals(sol, true);
  for (unsigned int i = 0; i < this->Nplayers; ++i) {
    double val = this->RespondSol(violSol, i, sol, true);
    if (val == GRB_INFINITY)
      return false;
    if (abs(val - objvals.at(i)) > tol) {
      violPlayer = i;
      return false;
    }
  }
  return true;
}

// EPEC stuff

void Game::EPEC::prefinalize()
/**
  @brief Empty function - optionally reimplementable in derived class
@details This function can be optionally implemented by
 the derived class. Code in this class will be run <i>before</i>
 calling Game::EPEC::finalize().
*/
{}

void Game::EPEC::postfinalize()
/**
  @brief Empty function - optionally reimplementable in derived class
@details This function can be optionally implemented by
 the derived class. Code in this class will be run <i>after</i>
 calling Game::EPEC::finalize().
*/
{}

void Game::EPEC::resetLCP() {
  /**
   * Resets the LCP objects to blank objects with no polyhedron added.
   * Useful in testing, or resolving a problem with a different algorithm.
   */
  BOOST_LOG_TRIVIAL(warning) << "Game::EPEC::resetLCP: resetting LCPs.";
  for (unsigned int i = 0; i < this->nCountr; ++i) {
    if (this->countries_LCP.at(i))
      this->countries_LCP.at(i).reset();
    this->countries_LCP.at(i) = std::unique_ptr<Game::LCP>(
        new LCP(this->env, *this->countries_LL.at(i).get()));
  }
}

void Game::EPEC::finalize()
/**
 * @brief Finalizes the creation of a Game::EPEC object.
 * @details Performs a bunch of job after all data for a Game::EPEC object are
 * given, namely.
 * Models::EPEC::computeLeaderLocations -	Adds the required dummy
 * variables to each leader's problem so that a game among the leaders can be
 * defined. Calls Game::EPEC::add_Dummy_Lead
 * 	-	Makes the market clearing constraint in each country. Calls
 */
{
  if (this->finalized)
    cerr << "Warning in Game::EPEC::finalize: Model already finalized\n";

  this->nCountr = this->getNcountries();
  /// Game::EPEC::prefinalize() can be overridden, and that code will run before
  /// calling Game::EPEC::finalize()
  this->prefinalize();

  try {
    this->convexHullVariables = std::vector<unsigned int>(this->nCountr, 0);
    this->Stats.feasiblePolyhedra = std::vector<unsigned int>(this->nCountr, 0);
    this->computeLeaderLocations(this->n_MCVar);
    // Initialize leader objective and country_QP
    this->LeadObjec = vector<shared_ptr<Game::QP_objective>>(nCountr);
    this->LeadObjec_ConvexHull =
        vector<shared_ptr<Game::QP_objective>>(nCountr);
    this->country_QP = vector<shared_ptr<Game::QP_Param>>(nCountr);
    this->countries_LCP = vector<unique_ptr<Game::LCP>>(nCountr);
    this->SizesWithoutHull = vector<unsigned int>(nCountr, 0);
    for (unsigned int i = 0; i < this->nCountr; i++) {
      this->add_Dummy_Lead(i);
      this->LeadObjec.at(i) = std::make_shared<Game::QP_objective>();
      this->LeadObjec_ConvexHull.at(i) = std::make_shared<Game::QP_objective>();
      this->make_obj_leader(i, *this->LeadObjec.at(i).get());
      this->countries_LCP.at(i) = std::unique_ptr<Game::LCP>(
          new LCP(this->env, *this->countries_LL.at(i).get()));
      this->SizesWithoutHull.at(i) = *this->LocEnds.at(i);
    }

  } catch (const char *e) {
    cerr << e << '\n';
    throw;
  } catch (string e) {
    cerr << "String in Game::EPEC::finalize : " << e << '\n';
    throw;
  } catch (GRBException &e) {
    cerr << "GRBException in Game::EPEC::finalize : " << e.getErrorCode()
         << ": " << e.getMessage() << '\n';
    throw;
  } catch (exception &e) {
    cerr << "Exception in Game::EPEC::finalize : " << e.what() << '\n';
    throw;
  }

  this->finalized = true;

  /// Game::EPEC::postfinalize() can be overridden, and that code will run after
  /// calling Game::EPEC::finalize()
  this->postfinalize();
}

void Game::EPEC::add_Dummy_Lead(
    const unsigned int i ///< The leader to whom dummy variables should be added
) {
  /// Adds dummy variables to the leader of an EPEC - useful after computing the
  /// convex hull.
  const unsigned int nEPECvars = this->nVarinEPEC;
  const unsigned int nThisCountryvars = *this->LocEnds.at(i);
  // this->Locations.at(i).at(Models::LeaderVars::End);

  if (nEPECvars < nThisCountryvars)
    throw string(
        "String in Game::EPEC::add_Dummy_Lead: Invalid variable counts " +
        to_string(nEPECvars) + " and " + to_string(nThisCountryvars));

  try {
    this->countries_LL.at(i).get()->addDummy(nEPECvars - nThisCountryvars);
  } catch (const char *e) {
    cerr << e << '\n';
    throw;
  } catch (string e) {
    cerr << "String in Game::EPEC::add_Dummy_All_Lead : " << e << '\n';
    throw;
  } catch (GRBException &e) {
    cerr << "GRBException in Game::EPEC::add_Dummy_All_Lead : "
         << e.getErrorCode() << ": " << e.getMessage() << '\n';
    throw;
  } catch (exception &e) {
    cerr << "Exception in Game::EPEC::add_Dummy_All_Lead : " << e.what()
         << '\n';
    throw;
  }
}

void Game::EPEC::computeLeaderLocations(const unsigned int addSpaceForMC) {
  this->LeaderLocations = vector<unsigned int>(this->nCountr);
  this->LeaderLocations.at(0) = 0;
  for (unsigned int i = 1; i < this->nCountr; i++) {
    this->LeaderLocations.at(i) =
        this->LeaderLocations.at(i - 1) + *this->LocEnds.at(i - 1);
  }
  this->nVarinEPEC =
      this->LeaderLocations.back() + *this->LocEnds.back() + addSpaceForMC;
}

void EPEC::get_x_minus_i(const arma::vec &x, const unsigned int &i,
                         arma::vec &solOther) const {
  const unsigned int nEPECvars = this->nVarinEPEC;
  const unsigned int nThisCountryvars = *this->LocEnds.at(i);
  const unsigned int nThisCountryHullVars = this->convexHullVariables.at(i);
  const unsigned int nConvexHullVars = std::accumulate(
      this->convexHullVariables.rbegin(), this->convexHullVariables.rend(), 0);

  solOther.zeros(nEPECvars -        // All variables in EPEC
                 nThisCountryvars - // Subtracting this country's variables,
                 // since we only want others'
                 nConvexHullVars + // We don't want any convex hull variables
                 nThisCountryHullVars); // We double subtracted our country's
  // convex hull vars

  for (unsigned int j = 0, count = 0, current = 0; j < this->nCountr; ++j) {
    if (i != j) {
      current = *this->LocEnds.at(j) - this->convexHullVariables.at(j);
      solOther.subvec(count, count + current - 1) =
          x.subvec(this->LeaderLocations.at(j),
                   this->LeaderLocations.at(j) + current - 1);
      count += current;
    }
    // We need to keep track of MC_vars also for this country
    solOther.at(solOther.n_rows - this->n_MCVar + j) =
        x.at(this->nVarinEPEC - this->n_MCVar + j);
  }
}

unique_ptr<GRBModel> Game::EPEC::Respond(const unsigned int i,
                                         const arma::vec &x) const {
  if (!this->finalized)
    throw string("Error in Game::EPEC::Respond: Model not finalized");

  if (i >= this->nCountr)
    throw string("Error in Game::EPEC::Respond: Invalid country number");

  arma::vec solOther;
  this->get_x_minus_i(x, i, solOther);
  return this->countries_LCP.at(i).get()->MPECasMILP(
      this->LeadObjec.at(i).get()->C, this->LeadObjec.at(i).get()->c, solOther,
      true);
}
double Game::EPEC::RespondSol(
    arma::vec &sol,      ///< [out] Optimal response
    unsigned int player, ///< Player whose optimal response is to be computed
    const arma::vec &x, ///< A vector of pure strategies (either for all players
    ///< or all other players
    const arma::vec &prevDev = {}
    //< [in] if any, the vector of previous deviations.
    ) const {
  /**
   * @brief Returns the optimal objective value that is obtainable for the
   * player @p player given the decision @p x of all other players.
   * @details
   * Calls Game::EPEC::Respond and obtains the unique_ptr to GRBModel of
   * best response by player @p player. Then solves the model and returns the
   * appropriate objective value.
   * @returns The optimal objective value for the player @p player.
   */
  auto model = this->Respond(player, x);
  const int status = model->get(GRB_IntAttr_Status);
  if (status == GRB_UNBOUNDED || status == GRB_OPTIMAL) {
    unsigned int Nx = this->countries_LCP.at(player)->getNcol();
    sol.zeros(Nx);
    for (unsigned int i = 0; i < Nx; ++i)
      sol.at(i) =
          model->getVarByName("x_" + to_string(i)).get(GRB_DoubleAttr_X);

    if (status == GRB_UNBOUNDED) {
      BOOST_LOG_TRIVIAL(warning) << "Game::EPEC::Respondsol: deviation is "
                                    "unbounded.";
      GRBLinExpr obj = 0;
      model->setObjective(obj);
      model->optimize();
      if (!prevDev.empty()) {
        BOOST_LOG_TRIVIAL(trace)
            << "Generating an improvement basing on the extreme ray.";
        // Fetch objective function coefficients
        GRBQuadExpr QuadObj = model->getObjective();
        arma::vec objcoeff;
        for (unsigned int i = 0; i < QuadObj.size(); ++i)
          objcoeff.at(i) = QuadObj.getCoeff(i);

        // Create objective function objects
        arma::vec objvalue = prevDev * objcoeff;
        arma::vec newobjvalue{0};
        double improved{false};

        // improve following the unbounded ray
        while (!improved) {
          for (unsigned int i = 0; i < Nx; ++i)
            sol.at(i) = sol.at(i) + model->getVarByName("x_" + to_string(i))
                                        .get(GRB_DoubleAttr_UnbdRay);
          newobjvalue = sol * objcoeff;
          if (newobjvalue.at(0) < objvalue.at(0))
            improved = true;
        }
        return newobjvalue.at(0);

      } else {
        return model->get(GRB_DoubleAttr_ObjVal);
      }
    }
    if (status == GRB_OPTIMAL) {
      return model->get(GRB_DoubleAttr_ObjVal);
    }
  } else {
    return GRB_INFINITY;
  }
  return GRB_INFINITY;
}

bool Game::EPEC::isSolved(unsigned int *countryNumber, arma::vec *ProfDevn,
                          double tol) const
/**
 * @briefs Checks if Game::EPEC is solved, else returns proof of unsolvedness.
 * @details
 * Analogous to Game::NashGame::isSolved but checks if the given Game::EPEC is
 * solved. If it is solved, then retruns true. If not, it returns the country
 * which has a profitable deviation in @p countryNumber and the profitable
 * deviation in @p ProfDevn. @p tol is the tolerance for the check. If the <i>
 * improved objective </i> after the deviation is less than @p tol, then it is
 * not considered as a profitable deviation.
 *
 * Thus we check if the given point is an @f$\epsilon@f$-equilibrium. Value of
 * @f$\epsilon @f$ can be chosen sufficiently close to 0.
 *
 * @warning Setting @p tol = 0 might even reject a real solution as not solved.
 * This is due to numerical issues arising from the LCP solver (Gurobi).
 */
{
  if (!this->nashgame)
    return false;
  if (!this->nashEq)
    return false;
  this->nashgame->isSolved(this->sol_x, *countryNumber, *ProfDevn);
  arma::vec objvals = this->nashgame->ComputeQPObjvals(this->sol_x, true);
  for (unsigned int i = 0; i < this->nCountr; ++i) {
    double val = this->RespondSol(*ProfDevn, i, this->sol_x);
    if (val == GRB_INFINITY)
      return false;
    if (abs(val - objvals.at(i)) > tol) {
      BOOST_LOG_TRIVIAL(trace)
          << "Game::EPEC::isSolved: found a deviation ("
          << abs(val - objvals.at(i)) << ")for player " << i
          << ".\nActual: " << objvals.at(i) << "\tOptimized: " << val;
      *countryNumber = i;
      // cout << "Proof - deviation: "<<*ProfDevn; //Sane
      // cout << "Current soln: "<<this->sol_x;
      return false;
    }
  }
  return true;
}

void Game::EPEC::make_country_QP(const unsigned int i)
/**
 * @brief Makes the Game::QP_Param corresponding to the @p i-th country.
 * @details
 *  - First gets the Game::LCP object from @p Game::EPEC::countries_LL and makes
 * a Game::QP_Param with this LCP as the lower level
 *  - This is achieved by calling LCP::makeQP and using the objective value
 * object in @p Game::EPEC::LeadObjec
 *  - Finally the locations are updated owing to the complete convex hull
 * calculated during the call to LCP::makeQP
 * @note Overloaded as Models::EPEC::make_country_QP()
 */
{
  // BOOST_LOG_TRIVIAL(info) << "Starting Convex hull computation of the country
  // "
  // << this->AllLeadPars[i].name << '\n';
  if (!this->finalized)
    throw string("Error in Game::EPEC::make_country_QP: Model not finalized");
  if (i >= this->nCountr)
    throw string(
        "Error in Game::EPEC::make_country_QP: Invalid country number");
  // if (!this->country_QP.at(i).get())
  {
    this->country_QP.at(i) = std::make_shared<Game::QP_Param>(this->env);
    const auto &origLeadObjec = *this->LeadObjec.at(i).get();

    this->LeadObjec_ConvexHull.at(i).reset(new Game::QP_objective{
        origLeadObjec.Q, origLeadObjec.C, origLeadObjec.c});

    this->countries_LCP.at(i)->makeQP(*this->LeadObjec_ConvexHull.at(i).get(),
                                      *this->country_QP.at(i).get());
    this->Stats.feasiblePolyhedra.at(i) =
        this->countries_LCP.at(i)->getFeasiblePolyhedra();
  }
}

void Game::EPEC::make_country_QP()
/**
 * @brief Makes the Game::QP_Param for all the countries
 * @details
 * Calls are made to Models::EPEC::make_country_QP(const unsigned int i) for
 * each valid @p i
 * @note Overloaded as EPEC::make_country_QP(unsigned int)
 * @todo manage removal of convexHull variables (eg,  convHullVarCount is
 * negative)
 */
{
  for (unsigned int i = 0; i < this->nCountr; ++i) {
    this->Game::EPEC::make_country_QP(i);
  }
  for (unsigned int i = 0; i < this->nCountr; ++i) {
    // LeadLocs &Loc = this->Locations.at(i);
    // Adjusting "stuff" because we now have new convHull variables
    unsigned int originalSizeWithoutHull = this->LeadObjec.at(i)->Q.n_rows;
    unsigned int convHullVarCount =
        this->LeadObjec_ConvexHull.at(i)->Q.n_rows - originalSizeWithoutHull;

    BOOST_LOG_TRIVIAL(trace)
        << "Game::EPEC::make_country_QP: Added " << convHullVarCount
        << " convex hull variables to QP #" << i;

    // Location details
    this->convexHullVariables.at(i) = convHullVarCount;
    // All other players' QP
    try {
      if (this->nCountr > 1) {
        for (unsigned int j = 0; j < this->nCountr; j++) {
          if (i != j) {
            this->country_QP.at(j)->addDummy(
                convHullVarCount, 0,
                this->country_QP.at(j)->getNx() -
                    this->n_MCVar); // The position to add parameters is towards
                                    // the end of all parameters, giving space
                                    // only for the n_MCVar number of market
                                    // clearing variables
          }
        }
      }
    } catch (const char *e) {
      cerr << e << '\n';
      throw;
    } catch (string e) {
      cerr << "String in Game::EPEC::make_country_QP : " << e << '\n';
      throw;
    } catch (GRBException &e) {
      cerr << "GRBException in Game::EPEC::make_country_QP : "
           << e.getErrorCode() << ": " << e.getMessage() << '\n';
      throw;
    } catch (exception &e) {
      cerr << "Exception in Game::EPEC::make_country_QP : " << e.what() << '\n';
      throw;
    }
  }
  this->updateLocs();
  this->computeLeaderLocations(this->n_MCVar);
}

bool Game::EPEC::getAllDevns(
    std::vector<arma::vec>
        &devns, ///< [out] The vector of deviations for all players
    const arma::vec &guessSol, ///< [in] The guess for the solution vector
    const std::vector<arma::vec>
        &prevDev //<[in] The previous vecrtor of deviations, if any exist.
    ) const
/**
 * @brief Given a potential solution vector, returns a profitable deviation (if
 * it exists) for all players. @param
 * @return a vector of computed deviations, which empty if at least one
 * deviation cannot be computed
 * @param prevDev can be empty
 * @todo Handle unbounded case
 */
{
  devns = std::vector<arma::vec>(this->nCountr);

  for (unsigned int i = 0; i < this->nCountr; ++i) { // For each country
    // If we cannot compute a deviation, it means model is infeasible!
    if (this->RespondSol(devns.at(i), i, guessSol, prevDev.at(i)) ==
        GRB_INFINITY)
      return false;
    // cout << "Game::EPEC::getAllDevns: devns(i): " <<devns.at(i);
  }
  return true;
}

unsigned int Game::EPEC::addDeviatedPolyhedron(
    const std::vector<arma::vec>
        &devns,       ///< devns.at(i) is a profitable deviation
                      ///< for the i-th country from the current this->sol_x
    bool &infeasCheck ///< Useful for the first iteration of iterativeNash. If
                      ///< true, at least one player has no polyhedron that can
                      ///< be added. In the first iteration, this translates to
                      ///< infeasability
    ) const {
  /**
   * Given a profitable deviation for each country, adds <i>a</i> polyhedron in
   * the feasible region of each country to the corresponding country's
   * Game::LCP object (this->countries_LCP.at(i)) 's vector of feasible
   * polyhedra.
   *
   * Naturally, this makes the inner approximation of the Game::LCP better, by
   * including one additional polyhedron.
   */

  infeasCheck = false;
  unsigned int added = 0;
  for (unsigned int i = 0; i < this->nCountr; ++i) { // For each country
    bool ret = false;
    if (!devns.at(i).empty())
      this->countries_LCP.at(i)->addPolyFromX(devns.at(i), ret);
    if (ret) {
      BOOST_LOG_TRIVIAL(trace)
          << "Game::EPEC::addDeviatedPolyhedron: added polyhedron for player "
          << i;
      ++added;
    } else {
      infeasCheck = true;
      BOOST_LOG_TRIVIAL(trace) << "Game::EPEC::addDeviatedPolyhedron: NO "
                                  "polyhedron added for player "
                               << i;
    }
  }
  return added;
}

bool Game::EPEC::addRandomPoly2All(unsigned int aggressiveLevel,
                                   bool stopOnSingleInfeasibility)
/**
 * Makes a call to to Game::LCP::addAPoly for each member in
 * Game::EPEC::countries_LCP and tries to add a polyhedron to get a better inner
 * approximation for the LCP. @p aggressiveLevel is the maximum number of
 * polyhedra it will try to add to each country. Setting it to an arbitrarily
 * high value will mimic complete enumeration.
 *
 * If @p stopOnSingleInfeasibility is true, then the function returns false and
 * aborts all operation as soon as it finds that it cannot add polyhedra to some
 * country. On the other hand if @p stopOnSingleInfeasibility is false, the
 * function returns false, only if it is not possible to add polyhedra to
 * <i>any</i> of the countries.
 * @returns true if successfully added the maximum possible number of polyhedra
 * not greater than aggressiveLevel.
 */
{
  BOOST_LOG_TRIVIAL(trace) << "Adding random polyhedra to countries";
  bool infeasible{true};
  for (unsigned int i = 0; i < this->nCountr; i++) {
    auto addedPolySet = this->countries_LCP.at(i)->addAPoly(
        aggressiveLevel, this->Stats.AlgorithmParam.addPolyMethod);
    if (stopOnSingleInfeasibility && addedPolySet.empty()) {
      BOOST_LOG_TRIVIAL(info)
          << "Game::EPEC::addRandomPoly2All: No Nash equilibrium. due to "
             "infeasibility of country "
          << i;
      return false;
    }
    if (!addedPolySet.empty())
      infeasible = false;
  }
  return !infeasible;
}

void Game::EPEC::iterativeNash() {

  // Set the initial point for all countries as 0 and solve the respective LCPs?
  this->sol_x.zeros(this->nVarinEPEC);

  bool solved = {false};
  bool addRandPoly{false};
  bool infeasCheck{false};
  std::vector<arma::vec> prevDevns(this->nCountr);
  this->Stats.numIteration = 0;
  if (this->Stats.AlgorithmParam.addPolyMethod == EPECAddPolyMethod::random) {
    for (unsigned int i = 0; i < this->nCountr; ++i) {
      long int seed = this->Stats.AlgorithmParam.addPolyMethodSeed < 0
                          ? chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count() +
                                42 + this->countries_LCP.at(i)->getNrow()
                          : this->Stats.AlgorithmParam.addPolyMethodSeed;
      this->countries_LCP.at(i)->addPolyMethodSeed = seed;
    }
  }
  std::chrono::high_resolution_clock::time_point initTime;
  if (this->Stats.AlgorithmParam.timeLimit > 0)
    initTime = std::chrono::high_resolution_clock::now();

  // Stay in this loop, till you find a Nash equilibrium or prove that there
  // does not exist a Nash equilibrium or you run out of time.
  while (!solved) {
    ++this->Stats.numIteration;
    BOOST_LOG_TRIVIAL(info) << "Game::EPEC::iterativeNash: Iteration "
                            << to_string(this->Stats.numIteration);
    if (addRandPoly) {
      BOOST_LOG_TRIVIAL(info)
          << "Game::EPEC::iterativeNash: using heuristical polyhedra selection";
      bool success =
          this->addRandomPoly2All(this->Stats.AlgorithmParam.aggressiveness,
                                  this->Stats.numIteration == 1);
      if (!success) {
        this->Stats.status = Game::EPECsolveStatus::nashEqNotFound;
        solved = true;
        return;
      }
    } else { // else we are in the case of finding deviations.
      unsigned int deviatedCountry{0};
      arma::vec countryDeviation{};
      if (this->isSolved(&deviatedCountry, &countryDeviation)) {
        this->Stats.status = Game::EPECsolveStatus::nashEqFound;
        solved = true;
        return;
      }
      // Vector of deviations for the countries
      std::vector<arma::vec> devns = std::vector<arma::vec>(this->nCountr);
      this->getAllDevns(devns, this->sol_x, prevDevns);
      prevDevns = devns;
      unsigned int addedPoly = this->addDeviatedPolyhedron(devns, infeasCheck);
      if (addedPoly == 0 && this->Stats.numIteration > 1) {
        BOOST_LOG_TRIVIAL(error)
            << " In Game::EPEC::iterativeNash: Not "
               "Solved, but no deviation? Error!\n This might be due to "
               "numerical issues (tollerances)";
        this->Stats.status = EPECsolveStatus::numerical;
        solved = true;
      }
      if (infeasCheck == true && this->Stats.numIteration == 1) {
        BOOST_LOG_TRIVIAL(error)
            << " In Game::EPEC::iterativeNash: Problem is infeasible";
        this->Stats.status = EPECsolveStatus::nashEqNotFound;
        solved = true;
        return;
      }
    }
    this->make_country_QP();

    // TimeLimit
    if (this->Stats.AlgorithmParam.timeLimit > 0) {
      const std::chrono::duration<double> timeElapsed =
          std::chrono::high_resolution_clock::now() - initTime;
      const double timeRemaining =
          this->Stats.AlgorithmParam.timeLimit - timeElapsed.count();
      addRandPoly = !this->computeNashEq(timeRemaining);
    } else {
      // No Time Limit
      addRandPoly = !this->computeNashEq();
    }
    if (addRandPoly)
      this->Stats.lostIntermediateEq++;
    // this->lcp->save("dat/LCP_alg.dat");
    // this->lcpmodel->write("dat/lcpmodel_alg.lp");
    for (unsigned int i = 0; i < this->nCountr; ++i) {
      BOOST_LOG_TRIVIAL(info)
          << "Country " << i << this->countries_LCP.at(i)->feas_detail_str();
    }
    // This might be reached when a NashEq is found, and need to be verified.
    // Anyway, we are over the timeLimit and we should stop
    if (this->Stats.AlgorithmParam.timeLimit > 0) {
      const std::chrono::duration<double> timeElapsed =
          std::chrono::high_resolution_clock::now() - initTime;
      const double timeRemaining =
          this->Stats.AlgorithmParam.timeLimit - timeElapsed.count();
      if (timeRemaining <= 0) {
        solved = false;
        this->Stats.status = Game::EPECsolveStatus::timeLimit;
        return;
      }
    }
  }
}

void ::Game::EPEC::make_country_LCP() {
  if (this->country_QP.front() == nullptr) {
    BOOST_LOG_TRIVIAL(error)
        << "Exception in Game::EPEC::make_country_LCP : no country QP has been "
           "made."
        << '\n';
    throw;
  }
  // Preliminary set up to get the LCP ready
  int Nvar =
      this->country_QP.front()->getNx() + this->country_QP.front()->getNy();
  arma::sp_mat MC(0, Nvar), dumA(0, Nvar);
  arma::vec MCRHS, dumb;
  MCRHS.zeros(0);
  dumb.zeros(0);
  this->make_MC_cons(MC, MCRHS);
  BOOST_LOG_TRIVIAL(trace) << "Game::EPEC::make_country_LCP(): Market Clearing "
                              "constraints are ready";
  this->nashgame = std::unique_ptr<Game::NashGame>(new Game::NashGame(
      this->env, this->country_QP, MC, MCRHS, 0, dumA, dumb));
  BOOST_LOG_TRIVIAL(trace)
      << "Game::EPEC::make_country_LCP(): NashGame is ready";
  this->lcp = std::unique_ptr<Game::LCP>(new Game::LCP(this->env, *nashgame));
  BOOST_LOG_TRIVIAL(trace) << "Game::EPEC::make_country_LCP(): LCP is ready";
  BOOST_LOG_TRIVIAL(trace)
      << "Game::EPEC::make_country_LCP(): indicators set to "
      << this->Stats.AlgorithmParam.indicators;
  this->lcp->useIndicators =
      this->Stats.AlgorithmParam.indicators; // Using indicator constraints

  this->lcpmodel = this->lcp->LCPasMIP(false);

  BOOST_LOG_TRIVIAL(trace) << *nashgame;
}

bool Game::EPEC::computeNashEq(
    double localTimeLimit ///< Allowed time limit to run this function
) {
  /**
   * Given that Game::EPEC::country_QP are all filled with a each country's
   * Game::QP_Param problem (either exact or approximate), computes the Nash
   * equilibrium.
   * @returns true if a Nash equilibrium is found
   */
  bool foundNash{false};
  // Make the Nash Game between countries
  BOOST_LOG_TRIVIAL(trace)
      << " Game::EPEC::computeNashEq: Making the Master LCP";
  this->make_country_LCP();
  BOOST_LOG_TRIVIAL(trace) << " Game::EPEC::computeNashEq: Made the Master LCP";
  if (localTimeLimit > 0) {
    this->lcpmodel->set(GRB_DoubleParam_TimeLimit, localTimeLimit);
  }
  if (this->Stats.AlgorithmParam.boundPrimals) {
    for (unsigned int c = 0; c < this->nashgame->getNprimals(); c++) {
      this->lcpmodel->getVarByName("x_" + to_string(c))
          .set(GRB_DoubleAttr_UB, this->Stats.AlgorithmParam.boundBigM);
    }
  }

  this->lcpmodel->optimize();
  this->lcpmodel->write("dat/anLCP.lp");
  this->Stats.wallClockTime += this->lcpmodel->get(GRB_DoubleAttr_Runtime);

  // Search just for a feasible point
  try { // Try finding a Nash equilibrium for the approximation
    foundNash =
        this->lcp->extractSols(this->lcpmodel.get(), sol_z, sol_x, true);
  } catch (GRBException &e) {
    BOOST_LOG_TRIVIAL(error)
        << "GRBException in Game::EPEC::computeNashEq : " << e.getErrorCode()
        << ": " << e.getMessage() << " ";
  }
  if (foundNash) { // If a Nash equilibrium is found, then update appropriately
    BOOST_LOG_TRIVIAL(info)
        << "Game::EPEC::computeNashEq: an equilibrium has been found.";
    this->nashEq = true;
    this->Stats.status = Game::EPECsolveStatus::nashEqFound;

  } else { // If not, then update accordingly
    BOOST_LOG_TRIVIAL(info)
        << "Game::EPEC::computeNashEq: no equilibrium has been found.";
    int status = this->lcpmodel->get(GRB_IntAttr_Status);
    if (status == GRB_TIME_LIMIT)
      this->Stats.status = Game::EPECsolveStatus::timeLimit;
    else
      this->Stats.status = Game::EPECsolveStatus::nashEqNotFound;
  }
  return foundNash;
}

bool Game::EPEC::warmstart(const arma::vec x) {

  if (x.size() < this->getnVarinEPEC()) {
    BOOST_LOG_TRIVIAL(error)
        << "Exception in Game::EPEC::warmstart: number of variables "
           "does not fit this instance.";
    throw;
  }
  if (!this->finalized) {
    BOOST_LOG_TRIVIAL(error)
        << "Exception in Game::EPEC::warmstart: EPEC is not finalized.";
    throw;
  }
  if (this->country_QP.front() == nullptr) {
    BOOST_LOG_TRIVIAL(warning)
        << "Game::EPEC::warmstart: Generating QP as of warmstart.";
  }

  this->sol_x = x;
  std::vector<arma::vec> devns = std::vector<arma::vec>(this->nCountr);
  std::vector<arma::vec> prevDevns = std::vector<arma::vec>(this->nCountr);
  this->getAllDevns(devns, this->sol_x, prevDevns);
  this->make_country_QP();

  unsigned int c;
  arma::vec devn;

  if (this->isSolved(&c, &devn))
    BOOST_LOG_TRIVIAL(warning) << "Game::EPEC::warmstart: "
                                  "The loaded solution is optimal.";
  else
    BOOST_LOG_TRIVIAL(warning)
        << "Game::EPEC::warmstart: "
           "The loaded solution is NOT optimal. Trying to repair.";
  /// @todo Game::EPEC::warmstart - to complete implementation?
  return true;
}

void Game::EPEC::findNashEq() {
  /**
   * @brief Computes Nash equilibrium using the algorithm set in
   * Game::EPEC::algorithm
   * @details
   * Checks the value of Game::EPEC::algorithm and delegates the task to
   * appropriate algorithm wrappers.
   */

  std::stringstream final_msg;
  if (!this->finalized)
    throw string("Error in Game::EPEC::iterativeNash: Object not yet "
                 "finalized. ");

  if (this->Stats.status != Game::EPECsolveStatus::unInitialized) {
    BOOST_LOG_TRIVIAL(error)
        << "Game::EPEC::findNashEq: a Nash Eq was "
           "already found. Calling this findNashEq might lead to errors!";
    this->resetLCP();
  }

  // Choosing the appropriate algorithm
  switch (this->Stats.AlgorithmParam.algorithm) {

  case Game::EPECalgorithm::innerApproximation:
    final_msg << "Inner approximation algorithm complete. ";
    this->iterativeNash();
    break;

  case Game::EPECalgorithm::fullEnumeration:
    final_msg << "Full enumeration algorithm complete. ";
    for (unsigned int i = 0; i < this->nCountr; ++i)
      this->countries_LCP.at(i)->EnumerateAll(true);
    this->make_country_QP();
    BOOST_LOG_TRIVIAL(trace)
        << "Game::EPEC::findNashEq: Starting fullEnumeration search";
    this->computeNashEq(this->Stats.AlgorithmParam.timeLimit);
    this->lcp->save("dat/LCP_enum.dat");
    this->lcpmodel->write("dat/lcpmodel_enum.lp");
    break;
  }
  // Handing EPECStatistics object to track performance of algorithm
  if (this->lcpmodel) {
    this->Stats.numVar = this->lcpmodel->get(GRB_IntAttr_NumVars);
    this->Stats.numConstraints = this->lcpmodel->get(GRB_IntAttr_NumConstrs);
    this->Stats.numNonZero = this->lcpmodel->get(GRB_IntAttr_NumNZs);
  } // Assigning appropriate status messages after solving

  switch (this->Stats.status) {
  case Game::EPECsolveStatus::nashEqNotFound:
    final_msg << "No Nash equilibrium exists. ";
    break;
  case Game::EPECsolveStatus::nashEqFound:
    final_msg << "Found a Nash equilibrium. ";

    break;
  case Game::EPECsolveStatus::timeLimit:
    final_msg << "Nash equilibrium not found. Time limit attained";
    break;
  case Game::EPECsolveStatus::numerical:
    final_msg << "Nash equilibrium not found. Numerical issues might affect "
                 "this result.";
    break;
  default:
    final_msg << "Nash equilibrium not found. Time limit attained";
    break;
  }
  BOOST_LOG_TRIVIAL(info) << "Game::EPEC::findNashEq: " << final_msg.str();
}

void Game::EPEC::setAlgorithm(Game::EPECalgorithm algorithm)
/**
 * Decides the algorithm to be used for solving the given instance of the
 * problem. The choice of algorithms are documented in Game::EPECalgorithm
 */
{
  this->Stats.AlgorithmParam.algorithm = algorithm;
}

unsigned int Game::EPEC::getPosition_LeadFoll(const unsigned int i,
                                              const unsigned int j) const {
  /**
   * Get the position of the j-th Follower variable in the i-th leader
   * Querying Game::EPEC::lcpmodel for x[return-value] variable gives the
   * appropriate variable
   */
  const auto LeaderStart = this->nashgame->getPrimalLoc(i);
  return LeaderStart + j;
}

unsigned int Game::EPEC::getPosition_LeadLead(const unsigned int i,
                                              const unsigned int j) const {
  /**
   * Get the position of the j-th Follower variable in the i-th leader
   * Querying Game::EPEC::lcpmodel for x[return-value] variable gives the
   * appropriate variable
   */
  const auto LeaderStart = this->nashgame->getPrimalLoc(i);
  return LeaderStart + this->countries_LCP.at(i)->getLStart() + j;
}

unsigned int Game::EPEC::getPosition_LeadFollPoly(const unsigned int i,
                                                  const unsigned int j,
                                                  const unsigned int k) const {
  /**
   * Get the position of the k-th follower variable of the i-th leader, in the
   * j-th feasible polyhedron.
   *
   * Indeed it should hold that @f$ j < @f$ Game::EPEC::getNPoly_Lead(i)
   */
  const auto LeaderStart = this->nashgame->getPrimalLoc(i);
  const auto FollPoly = this->countries_LCP.at(i)->conv_PolyPosition(k);
  return LeaderStart + FollPoly + j;
}

unsigned int Game::EPEC::getPosition_LeadLeadPoly(const unsigned int i,
                                                  const unsigned int j,
                                                  const unsigned int k) const {
  /**
   * Get the position of the k-th leader variable of the i-th leader, in the
   * j-th feasible polyhedron.
   *
   * Indeed it should hold that @f$ j < @f$ Game::EPEC::getNPoly_Lead(i)
   */
  const auto LeaderStart = this->nashgame->getPrimalLoc(i);
  const auto FollPoly = this->countries_LCP.at(i)->conv_PolyPosition(k);
  return LeaderStart + FollPoly + this->countries_LCP.at(i)->getLStart() + j;
}

unsigned int Game::EPEC::getNPoly_Lead(const unsigned int i) const {
  /**
   * Get the number of polyhedra used in the inner approximation of the feasible
   * region of the i-th leader.*
   */
  return this->countries_LCP.at(i)->conv_Npoly();
}

unsigned int Game::EPEC::getPosition_Probab(const unsigned int i,
                                            const unsigned int k) const {
  /**
   * Get the position of the probability associated with the k-th polyhedron
   * (k-th pure strategy) of the i-th leader. However, if the leader has an
   * inner approximation with exactly 1 polyhedron, it returns 0;
   */
  const auto PolyProbab = this->countries_LCP.at(i)->conv_PolyWt(k);
  if (PolyProbab == 0)
    return 0;
  const auto LeaderStart = this->nashgame->getPrimalLoc(i);
  return LeaderStart + PolyProbab;
}

bool Game::EPEC::isPureStrategy(const unsigned int i, const double tol) const {
  /**
   * Checks if the returned strategy leader is a pure strategy for the leader i.
   * The strategy is considered a pure strategy, if it is played with a
   * probability greater than 1 - tol;
   */
  const unsigned int nPoly = this->getNPoly_Lead(i);
  for (unsigned int j = 0; j < nPoly; j++) {
    const double probab = this->getVal_Probab(i, j);
    if (probab > 1 - tol) // Current Strategy is a pure strategy!
      return true;
  }
  return false;
}

std::vector<unsigned int> Game::EPEC::mixedStratPoly(const unsigned int i,
                                                     const double tol) const
/**
 * Returns the indices of polyhedra feasible for the leader, from which
 * strategies are played with probability greater than tol.
 */
{
  std::vector<unsigned int> polys{};
  const unsigned int nPoly = this->getNPoly_Lead(i);
  for (unsigned int j = 0; j < nPoly; j++) {
    const double probab = this->getVal_Probab(i, j);
    if (probab > tol)
      polys.push_back(j);
  }
  cout << "\n";
  return polys;
}

double Game::EPEC::getVal_Probab(const unsigned int i,
                                 const unsigned int k) const {
  const unsigned int varname{this->getPosition_Probab(i, k)};
  if (varname == 0)
    return 1;
  return this->lcpmodel->getVarByName("x_" + std::to_string(varname))
      .get(GRB_DoubleAttr_X);
}

double Game::EPEC::getVal_LeadFoll(const unsigned int i,
                                   const unsigned int j) const {
  if (!this->lcpmodel)
    throw std::string("Error in Game::EPEC::getVal_LeadFoll: "
                      "Game::EPEC::lcpmodel not made and solved");
  return this->lcpmodel
      ->getVarByName("x_" + to_string(this->getPosition_LeadFoll(i, j)))
      .get(GRB_DoubleAttr_X);
}

double Game::EPEC::getVal_LeadLead(const unsigned int i,
                                   const unsigned int j) const {
  if (!this->lcpmodel)
    throw std::string("Error in Game::EPEC::getVal_LeadLead: "
                      "Game::EPEC::lcpmodel not made and solved");
  return this->lcpmodel
      ->getVarByName("x_" + to_string(this->getPosition_LeadLead(i, j)))
      .get(GRB_DoubleAttr_X);
}

double Game::EPEC::getVal_LeadFollPoly(const unsigned int i,
                                       const unsigned int j,
                                       const unsigned int k,
                                       const double tol) const {
  if (!this->lcpmodel)
    throw std::string("Error in Game::EPEC::getVal_LeadFollPoly: "
                      "Game::EPEC::lcpmodel not made and solved");
  const double probab = this->getVal_Probab(i, k);
  if (probab > 1 - tol)
    return this->getVal_LeadFoll(i, j);
  else
    return this->lcpmodel
               ->getVarByName(
                   "x_" + to_string(this->getPosition_LeadFollPoly(i, j, k)))
               .get(GRB_DoubleAttr_X) /
           probab;
}

double Game::EPEC::getVal_LeadLeadPoly(const unsigned int i,
                                       const unsigned int j,
                                       const unsigned int k,
                                       const double tol) const {
  if (!this->lcpmodel)
    throw std::string("Error in Game::EPEC::getVal_LeadLeadPoly: "
                      "Game::EPEC::lcpmodel not made and solved");
  const double probab = this->getVal_Probab(i, k);
  if (probab > 1 - tol)
    return this->getVal_LeadLead(i, j);
  else
    return this->lcpmodel
               ->getVarByName(
                   "x_" + to_string(this->getPosition_LeadLeadPoly(i, j, k)))
               .get(GRB_DoubleAttr_X) /
           probab;
}

std::string std::to_string(const Game::EPECsolveStatus st) {
  switch (st) {
  case EPECsolveStatus::nashEqNotFound:
    return string("NO_NASH_EQ_FOUND");
  case EPECsolveStatus::nashEqFound:
    return string("NASH_EQ_FOUND");
  case EPECsolveStatus::timeLimit:
    return string("TIME_LIMIT");
  case EPECsolveStatus::unInitialized:
    return string("UNINITIALIZED");
  case EPECsolveStatus::numerical:
    return string("NUMERICAL_ISSUES");
  default:
    return string("UNKNOWN_STATUS_") + to_string(static_cast<int>(st));
  }
}
std::string std::to_string(const Game::EPECalgorithm al) {
  switch (al) {
  case EPECalgorithm::fullEnumeration:
    return string("fullEnumeration");
  case EPECalgorithm::innerApproximation:
    return string("innerApproximation");
  default:
    return string("UNKNOWN_ALGORITHM_") + to_string(static_cast<int>(al));
  }
}
std::string std::to_string(const Game::EPECAddPolyMethod add) {
  switch (add) {
  case EPECAddPolyMethod::sequential:
    return string("sequential");
  case EPECAddPolyMethod::reverse_sequential:
    return string("reverse_sequential");
  case EPECAddPolyMethod::random:
    return string("random");
  default:
    return string("UNKNOWN_ALGORITHM_") + to_string(static_cast<int>(add));
  }
}
std::string std::to_string(const Game::EPECAlgorithmParams al) {
  std::stringstream ss;
  ss << "Algorithm: " << to_string(al.algorithm) << '\n';
  if (al.algorithm == Game::EPECalgorithm::innerApproximation) {
    ss << "Aggressiveness: " << al.aggressiveness << '\n';
    ss << "AddPolyMethod: " << to_string(al.addPolyMethod) << '\n';
  }
  ss << "Time Limit: " << al.timeLimit << '\n';
  ss << "Indicators: " << std::boolalpha << al.indicators;

  return ss.str();
}
