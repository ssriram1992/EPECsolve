#include "games.h"
#include <algorithm>
#include <armadillo>
#include <array>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <memory>

using namespace std;
using namespace Utils;

bool Game::isZero(arma::mat M, double tol) {
  return (arma::min(arma::min(abs(M))) <= tol);
}

bool Game::isZero(arma::sp_mat M, double tol) {
  return (arma::min(arma::min(abs(M))) <= tol);
}
// bool Game::isZero(arma::vec M, double tol)
// {
// return(arma::min(abs(M)) <= tol);
// }

template <class T> ostream &operator<<(ostream &ost, vector<T> v) {
  for (auto elem : v)
    ost << elem << " ";
  ost << '\n';
  return ost;
}

template <class T, class S> ostream &operator<<(ostream &ost, pair<T, S> p) {
  ost << "<" << p.first << ", " << p.second << ">";
  return ost;
}

void Game::print(const perps &C) {
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
  this->getQ().save(filename + "_Q.txt", arma::file_type::arma_ascii);
  this->getC().save(filename + "_C.txt", arma::file_type::arma_ascii);
  this->getA().save(filename + "_A.txt", arma::file_type::arma_ascii);
  this->getB().save(filename + "_B.txt", arma::file_type::arma_ascii);
  this->getc().save(filename + "_c.txt", arma::file_type::arma_ascii);
  this->getb().save(filename + "_b.txt", arma::file_type::arma_ascii);
}

void Game::QP_Param::write(string filename, bool append) const {
  // this->MP_Param::write(filename, append);
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
      A = arma::join_rows(A_temp, A.cols(position, A.n_cols - 1));
    }
    if (vars || pars) {
      C = resize_patch(C, this->Ny, C.n_cols);
      arma::sp_mat C_temp = arma::join_rows(
          C.cols(0, position - 1), arma::zeros<arma::sp_mat>(this->Ny, pars));
      C = arma::join_rows(C_temp, C.cols(position, C.n_cols - 1));
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

bool Game::MP_Param::dataCheck(
    bool forcesymm ///< Check if MP_Param::Q is symmetric
    ) const
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

unique_ptr<GRBModel>
Game::QP_Param::solveFixed(arma::vec x ///< Other players' decisions
                           )
/**
 * Given a value for the parameters @f$x@f$ in the definition of QP_Param, solve
 * the parameterized quadratic program to  optimality.
 *
 * In terms of game theory, this can be viewed as <i>the best response</i> for a
 * set of decisions by other players.
 *
 */
{
  this->make_yQy();
  /// @throws GRBException if argument vector size is not compatible with the
  /// Game::QP_Param definition.
  if (x.size() != this->Nx)
    throw "Invalid argument size: " + to_string(x.size()) +
        " != " + to_string(Nx);
  /// @warning Creates a GRBModel using dynamic memory. Should be freed by the
  /// caller.
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
    throw string("Inconsistent data for KKT of Game::QP_Param");
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
  if (y.n_rows != this->getNy())
    throw string("Error in QP_Param::computeObjective: Invalid size of y");
  if (x.n_rows != this->getNx())
    throw string("Error in QP_Param::computeObjective: Invalid size of x");
  if (checkFeas) {
    arma::vec slack = A * x + B * y - b;
    if (slack.n_rows) // if infeasible
      if (slack.max() >= tol)
        return -GRB_INFINITY;
    if (y.min() <= -tol) // if infeasible
      return -GRB_INFINITY;
  }
  arma::vec obj = 0.5 * y.t() * Q * y + (C * x).t() * y + c.t() * y;
  return obj(0);
}

void Game::QP_Param::save(string filename, bool erase) const {
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
   * @brief Loads the @p QP_Param object stored in a file.  Before calling this
   * function, use the constructor QP_Param::QP_Param(GRBEnv *env) to
   * initialize.
   * @details Loads the @p QP_Param object stored in a file.  Before calling
   * this function, use the constructor QP_Param::QP_Param(GRBEnv *env) to
   * initialize. Example usage:
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
 */
/**
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
  this->Players = Players;
  this->Nplayers = Players.size();
  this->MarketClearing = MC;
  this->MCRHS = MCRHS;
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

  /*
  if (VERBOSE) {
      cout << "Primals: ";
      for (unsigned int i = 0; i < Nplayers; i++) cout << primal_position.at(i)
  << " "; cout << "---MC_Dual:" << MC_dual_position << "---Leader: " <<
  Leader_position << "Duals: "; for (unsigned int i = 0; i < Nplayers + 1; i++)
  cout << dual_position.at(i) << " "; cout << '\n';
  }
  */
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
   *
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
  //

  for (unsigned int i = 0; i < Nplayers; i++) {
    // cout << "-----Player " << i << '\n';
    this->Players[i]->KKT(Mi[i], Ni[i], qi[i]);
    unsigned int Nprim, Ndual;
    Nprim = this->Players[i]->getNy();
    Ndual = this->Players[i]->getA().n_rows;
    // Adding the primal equations
    // Region 1 in Formulate LCP.ipe
    if (i > 0) { // For the first player, no need to add anything 'before' 0-th
                 // position
      // cout << "Region 1" << '\n';
      // cout << "\tM(" << this->primal_position.at(i) << "," << 0 << "," <<
      // this->primal_position.at(i + 1) - 1
      // << "-"
      // << this->primal_position.at(i) - 1 << ")" << '\n';
      // cout << "\t(" << 0 << "," << 0 << "-" << Nprim - 1 << "," <<
      // this->primal_position.at(i) - 1 << ")" << '\n';
      M.submat(this->primal_position.at(i), 0,
               this->primal_position.at(i + 1) - 1,
               this->primal_position.at(i) - 1) =
          Ni[i].submat(0, 0, Nprim - 1, this->primal_position.at(i) - 1);
    }
    // Region 2 in Formulate LCP.ipe
    // cout << "Region 2" << '\n';
    // cout << "\tM(" << this->primal_position.at(i) << "," <<
    // this->primal_position.at(i) << "-"
    // << this->primal_position.at(i + 1) - 1 << "-" <<
    // this->primal_position.at(i + 1) - 1 << ")" << '\n'; cout << "\t(" << 0 <<
    // "," << 0 << "-" << Nprim - 1 << "," << Nprim - 1 << ")" << '\n';
    M.submat(this->primal_position.at(i), this->primal_position.at(i),
             this->primal_position.at(i + 1) - 1,
             this->primal_position.at(i + 1) - 1) =
        Mi[i].submat(0, 0, Nprim - 1, Nprim - 1);
    // Region 3 in Formulate LCP.ipe
    if (this->primal_position.at(i + 1) != this->dual_position.at(0)) {
      // cout << "Region 3" << '\n';
      // cout << "\tM(" << this->primal_position.at(i) << "," <<
      // this->primal_position.at(i + 1) << "-"
      // << this->primal_position.at(i + 1) - 1 << "-" <<
      // this->dual_position.at(0) - 1 << ")" << '\n'; cout << "\t(" << 0 << ","
      // << this->primal_position.at(i) << "-" << Nprim - 1 << "," <<
      // Ni[i].n_cols - 1
      // << ")"
      // << '\n';
      M.submat(this->primal_position.at(i), this->primal_position.at(i + 1),
               this->primal_position.at(i + 1) - 1,
               this->dual_position.at(0) - 1) =
          Ni[i].submat(0, this->primal_position.at(i), Nprim - 1,
                       Ni[i].n_cols - 1);
    }
    // Region 4 in Formulate LCP.ipe
    if (this->dual_position.at(i) != this->dual_position.at(i + 1)) {
      // cout << "Region 4" << '\n';
      // cout << "\tM(" << this->primal_position.at(i) << "," <<
      // this->dual_position.at(i) << "-"
      // << this->primal_position.at(i + 1) - 1 << "-" <<
      // this->dual_position.at(i + 1) << ")" << '\n'; cout << "\t(" << 0 << ","
      // << Nprim - 1 << "-" << Nprim - 1 << "," << Nprim + Ndual - 1 << ")" <<
      // '\n';
      M.submat(this->primal_position.at(i), this->dual_position.at(i),
               this->primal_position.at(i + 1) - 1,
               this->dual_position.at(i + 1) - 1) =
          Mi[i].submat(0, Nprim, Nprim - 1, Nprim + Ndual - 1);
    }
    // RHS
    q.subvec(this->primal_position.at(i), this->primal_position.at(i + 1) - 1) =
        qi[i].subvec(0, Nprim - 1);
    for (unsigned int j = this->primal_position.at(i);
         j < this->primal_position.at(i + 1); j++)
      Compl.push_back({j, j});
    // Adding the dual equations
    // Region 5 in Formulate LCP.ipe
    if (Ndual > 0) {
      if (i > 0) // For the first player, no need to add anything 'before' 0-th
                 // position
        M.submat(this->dual_position.at(i) - n_LeadVar, 0,
                 this->dual_position.at(i + 1) - n_LeadVar - 1,
                 this->primal_position.at(i) - 1) =
            Ni[i].submat(Nprim, 0, Ni[i].n_rows - 1,
                         this->primal_position.at(i) - 1);
      // Region 6 in Formulate LCP.ipe
      M.submat(this->dual_position.at(i) - n_LeadVar,
               this->primal_position.at(i),
               this->dual_position.at(i + 1) - n_LeadVar - 1,
               this->primal_position.at(i + 1) - 1) =
          Mi[i].submat(Nprim, 0, Nprim + Ndual - 1, Nprim - 1);
      // Region 7 in Formulate LCP.ipe
      M.submat(this->dual_position.at(i) - n_LeadVar,
               this->primal_position.at(i + 1),
               this->dual_position.at(i + 1) - n_LeadVar - 1,
               this->dual_position.at(0) - 1) =
          Ni[i].submat(Nprim, this->primal_position.at(i), Ni[i].n_rows - 1,
                       Ni[i].n_cols - 1);
      // Region 8 in Formulate LCP.ipe
      M.submat(this->dual_position.at(i) - n_LeadVar, this->dual_position.at(i),
               this->dual_position.at(i + 1) - n_LeadVar - 1,
               this->dual_position.at(i + 1) - 1) =
          Mi[i].submat(Nprim, Nprim, Nprim + Ndual - 1, Nprim + Ndual - 1);
      // RHS
      q.subvec(this->dual_position.at(i) - n_LeadVar,
               this->dual_position.at(i + 1) - n_LeadVar - 1) =
          qi[i].subvec(Nprim, qi[i].n_rows - 1);
      for (unsigned int j = this->dual_position.at(i) - n_LeadVar;
           j < this->dual_position.at(i + 1) - n_LeadVar; j++)
        Compl.push_back({j, j + n_LeadVar});
    }
  }
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
    const arma::vec &x, ///< A vector of pure strategies (either for all players
                        ///< or all other players)
    bool fullvec        ///< Is @p x strategy of all players?
    ) const
/**
 * @brief Given the decision of other players, find the optimal response for
 * player in position @p player
 * @detail
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
    if (nEnd < x.n_rows)
      solOther.subvec(nStart, nVar + nStart - nEnd - 1) =
          x.subvec(nEnd,
                   nVar - 1); // Discard any dual variables in x
  } else {
    solOther.zeros(nVar + nEnd - nStart);
    solOther = x.subvec(0, nVar + nEnd - nStart -
                               1); // Discard any dual variables in x
  }

  return this->Players.at(player)->solveFixed(solOther);
}

double Game::NashGame::RespondSol(
    arma::vec &sol,
    unsigned int player, ///< Player whose optimal response is to be computed
    const arma::vec &x, ///< A vector of pure strategies (either for all players
                        ///< or all other players)
    bool fullvec        ///< Is @p x strategy of all players?
    ) const {
  auto model = this->Respond(player, x, fullvec);
  unsigned int Nx =
      this->primal_position.at(player + 1) - this->primal_position.at(player);
  sol.zeros(Nx);
  for (unsigned int i = 0; i < Nx; ++i)
    sol.at(i) = model->getVarByName("y_" + to_string(i)).get(GRB_DoubleAttr_X);

  return model->get(GRB_DoubleAttr_ObjVal);
}

arma::vec Game::NashGame::ComputeQPObjvals(const arma::vec &x,
                                           bool checkFeas) const {
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
    if (nStart > 0)
      x_minus_i.subvec(0, nStart - 1) = x.subvec(0, nStart - 1);
    if (nEnd < x.n_rows)
      x_minus_i.subvec(nStart, nVar + nStart - nEnd - 1) =
          x.subvec(nEnd,
                   nVar - 1); // Discard any dual variables in x

    x_i = x.subvec(nStart, nEnd - 1);

    vals.at(i) =
        this->Players.at(i)->computeObjective(x_i, x_minus_i, checkFeas);
  }

  return vals;
}

bool Game::NashGame::isSolved(const arma::vec &sol, unsigned int &violPlayer,
                              arma::vec &violSol, double tol) const {
  arma::vec objvals = this->ComputeQPObjvals(sol, true);
  for (unsigned int i = 0; i < this->Nplayers; ++i) {
    double val = this->RespondSol(violSol, i, sol, true);
    if (abs(val - objvals.at(i)) > tol) {
      violPlayer = i;
      return false;
    }
  }
  return true;
}

// EPEC stuff

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

  /// Game::EPEC::prefinalize() can be overridden, and that code will run before
  /// calling Game::EPEC::finalize()
  this->prefinalize();

  try {
    this->computeLeaderLocations(this->n_MCVar);
    // Initialize leader objective and country_QP
    this->LeadObjec = vector<shared_ptr<Game::QP_objective>>(nCountr);
    this->country_QP = vector<shared_ptr<Game::QP_Param>>(nCountr);
    for (unsigned int i = 0; i < this->nCountr; i++) {
      this->add_Dummy_Lead(i);
      this->LeadObjec.at(i) = std::make_shared<Game::QP_objective>();
      this->make_obj_leader(i, *this->LeadObjec.at(i).get());
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

void Game::EPEC::add_Dummy_Lead(const unsigned int i) {

  const unsigned int nEPECvars = this->nVarinEPEC;
  const unsigned int nThisCountryvars = *this->LocEnds.at(i);
  // this->Locations.at(i).at(Models::LeaderVars::End);

  try {
    this->countries_LL.at(i).get()->addDummy(nEPECvars - nThisCountryvars);
  } catch (const char *e) {
    cerr << e << '\n';
    throw;
  } catch (string e) {
    cerr << "String in Models::EPEC::add_Dummy_All_Lead : " << e << '\n';
    throw;
  } catch (GRBException &e) {
    cerr << "GRBException in Models::EPEC::add_Dummy_All_Lead : "
         << e.getErrorCode() << ": " << e.getMessage() << '\n';
    throw;
  } catch (exception &e) {
    cerr << "Exception in Models::EPEC::add_Dummy_All_Lead : " << e.what()
         << '\n';
    throw;
  }
}

void Game::EPEC::computeLeaderLocations(const unsigned int addSpaceForMC) {
  this->LeaderLocations = vector<unsigned int>(this->nCountr);
  this->LeaderLocations.at(0) = 0;
  for (unsigned int i = 1; i < this->nCountr; i++)
    this->LeaderLocations.at(i) =
        this->LeaderLocations.at(i - 1) + *this->LocEnds.at(i);

  this->nVarinEPEC =
      this->LeaderLocations.back() + *this->LocEnds.back() + addSpaceForMC;
}

unique_ptr<GRBModel> Game::EPEC::Respond(const unsigned int i,
                                         const arma::vec &x) const {
  if (!this->finalized)
    throw string("Error in Models::EPEC::Respond: Model not finalized");

  if (i >= this->nCountr)
    throw string("Error in Models::EPEC::Respond: Invalid country number");

  const unsigned int nEPECvars = this->nVarinEPEC;
  const unsigned int nThisCountryvars = *this->LocEnds.at(i);
  // this->Locations.at(i).at(Models::LeaderVars::End);

  if (x.n_rows != nEPECvars - nThisCountryvars)
    throw string("Error in Models::EPEC::Respond: Invalid parametrization");

  return this->country_QP.at(i).get()->solveFixed(x);
}

bool Game::EPEC::isSolved(unsigned int *countryNumber,
                          arma::vec *ProfDevn) const
/**
 * @todo Implementation to be done.
 */
{
  BOOST_LOG_TRIVIAL(fatal) << "NOT YET IMPLEMENTED";
  return false;
}

void Game::EPEC::make_country_QP(const unsigned int i)
/**
 * @brief Makes the Game::QP_Param corresponding to the @p i-th country.
 * @details
 *  - First gets the Game::LCP object from @p countries_LL and makes a QP with
 * this LCP as the lower level
 *  - This is achieved by calling LCP::makeQP and using the objective value
 * object in @p LeadObjec
 *  - Finally the locations are updated owing to the complete convex hull
 * calculated during the call to LCP::makeQP
 * @note Overloaded as EPEC::make_country_QP()
 * @todo where is the error?
 */
{
  // BOOST_LOG_TRIVIAL(info) << "Starting Convex hull computation of the country
  // "
  // << this->AllLeadPars[i].name << '\n';
  if (!this->finalized)
    throw string("Error in Models::EPEC::make_country_QP: Model not finalized");
  if (i >= this->nCountr)
    throw string(
        "Error in Models::EPEC::make_country_QP: Invalid country number");
  if (!this->country_QP.at(i).get()) {
    Game::LCP Player_i_LCP =
        Game::LCP(this->env, *this->countries_LL.at(i).get());
    this->country_QP.at(i) = std::make_shared<Game::QP_Param>(this->env);
    Player_i_LCP.makeQP(*this->LeadObjec.at(i).get(),
                        *this->country_QP.at(i).get());
    this->Stats.feasiblePolyhedra.push_back(
        Player_i_LCP.getFeasiblePolyhedra());
  }
}

void Game::EPEC::findNashEq() {
  if (this->country_QP.front() != nullptr) {

    int Nvar =
        this->country_QP.front()->getNx() + this->country_QP.front()->getNy();
    arma::sp_mat MC(0, Nvar), dumA(0, Nvar);
    arma::vec MCRHS, dumb;
    MCRHS.zeros(0);
    dumb.zeros(0);
    this->make_MC_cons(MC, MCRHS);
    this->nashgame = std::unique_ptr<Game::NashGame>(new Game::NashGame(
        this->env, this->country_QP, MC, MCRHS, 0, dumA, dumb));
    lcp = std::unique_ptr<Game::LCP>(new Game::LCP(this->env, *nashgame));
    // Using indicator constraints
    lcp->useIndicators = this->indicators;

    this->lcpmodel = lcp->LCPasMIP(false);
    Nvar = nashgame->getNprimals() + nashgame->getNduals() +
           nashgame->getNshadow() + nashgame->getNleaderVars();
    // if (VERBOSE) {
    // lcpmodel->write("dat/debug/NashLCP.lp");
    // this->nashgame->write("dat/debug/NashGame", false, true);
    BOOST_LOG_TRIVIAL(trace) << *nashgame;
    // }

    this->Stats.numVar = lcpmodel->get(GRB_IntAttr_NumVars);
    this->Stats.numConstraints = lcpmodel->get(GRB_IntAttr_NumConstrs);
    this->Stats.numNonZero = lcpmodel->get(GRB_IntAttr_NumNZs);
    if (this->timeLimit > 0) {
      BOOST_LOG_TRIVIAL(warning)
          << "Time limit set: " << this->Game::EPEC::timeLimit;
      this->lcpmodel->set(GRB_DoubleParam_TimeLimit, this->timeLimit);
    }
    lcpmodel->optimize();
    this->Stats.wallClockTime = lcpmodel->get(GRB_DoubleAttr_Runtime);
    this->sol_x.zeros(Nvar);
    this->sol_z.zeros(Nvar);
    unsigned int temp;
    int status = lcpmodel->get(GRB_IntAttr_Status);
    if (status == GRB_OPTIMAL || status == GRB_SUBOPTIMAL ||
        status == GRB_SOLUTION_LIMIT) {
      try {
        for (unsigned int i = 0; i < (unsigned int)Nvar; i++) {
          this->sol_x(i) =
              lcpmodel->getVarByName("x_" + to_string(i)).get(GRB_DoubleAttr_X);
          this->sol_z(i) =
              lcpmodel->getVarByName("z_" + to_string(i)).get(GRB_DoubleAttr_X);
          temp = i;
        }

      } catch (GRBException &e) {
        cerr << "GRBException in Game::EPEC::findNashEq : " << e.getErrorCode()
             << ": " << e.getMessage() << " " << temp << '\n';
      }
      this->Stats.status = 1;
    } else {
      if (status == GRB_TIME_LIMIT) {
        this->Stats.status = 2;
        cerr << "Game::EPEC::findNashEq: no nash equilibrium found "
                "(timeLimit)."
             << '\n';
      } else
        cerr << "Game::EPEC::findNashEq: no nash equilibrium found." << '\n';
    }

  } else {
    cerr << "Exception in Game::EPEC::findNashEq : no country QP has been "
            "made."
         << '\n';
    throw;
  }
}
