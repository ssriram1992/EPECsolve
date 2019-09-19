#pragma once

// #include"epecsolve.h"
#include "lcptolp.h"
#include <armadillo>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>

using namespace std;
using namespace Game;

template <class T> ostream &operator<<(ostream &ost, std::vector<T> v) {
  for (auto elem : v)
    ost << elem << " ";
  ost << '\n';
  return ost;
}

template <class T, class S> ostream &operator<<(ostream &ost, pair<T, S> p) {
  ost << "<" << p.first << ", " << p.second << ">";
  return ost;
}

namespace Game {
bool isZero(arma::mat M, double tol = 1e-6);

bool isZero(arma::sp_mat M, double tol = 1e-6);

// bool isZero(arma::vec M, double tol = 1e-6);
///@brief struct to handle the objective params of MP_Param/QP_Param
///@details Refer QP_Param class for what Q, C and c mean.
typedef struct QP_objective {
  arma::sp_mat Q, C;
  arma::vec c;
} QP_objective;
///@brief struct to handle the constraint params of MP_Param/QP_Param
///@details Refer QP_Param class for what A, B and b mean.
typedef struct QP_constraints {
  arma::sp_mat A, B;
  arma::vec b;
} QP_constraints;

///@brief class to handle parameterized mathematical programs(MP)
class MP_Param {
protected:
  // Data representing the parameterized QP
  arma::sp_mat Q, A, B, C;
  arma::vec c, b;

  // Object for sizes and integrity check
  unsigned int Nx, Ny, Ncons;

  unsigned int size();

  bool dataCheck(bool forcesymm = true) const;

public:
  // Default constructors
  MP_Param() = default;

  MP_Param(const MP_Param &M) = default;

  // Getters and setters
  virtual inline arma::sp_mat getQ() const final {
    return this->Q;
  } ///< Read-only access to the private variable Q
  virtual inline arma::sp_mat getC() const final {
    return this->C;
  } ///< Read-only access to the private variable C
  virtual inline arma::sp_mat getA() const final {
    return this->A;
  } ///< Read-only access to the private variable A
  virtual inline arma::sp_mat getB() const final {
    return this->B;
  } ///< Read-only access to the private variable B
  virtual inline arma::vec getc() const final {
    return this->c;
  } ///< Read-only access to the private variable c
  virtual inline arma::vec getb() const final {
    return this->b;
  } ///< Read-only access to the private variable b
  virtual inline unsigned int getNx() const final {
    return this->Nx;
  } ///< Read-only access to the private variable Nx
  virtual inline unsigned int getNy() const final {
    return this->Ny;
  } ///< Read-only access to the private variable Ny

  virtual inline MP_Param &setQ(const arma::sp_mat &Q) final {
    this->Q = Q;
    return *this;
  } ///< Set the private variable Q
  virtual inline MP_Param &setC(const arma::sp_mat &C) final {
    this->C = C;
    return *this;
  } ///< Set the private variable C
  virtual inline MP_Param &setA(const arma::sp_mat &A) final {
    this->A = A;
    return *this;
  } ///< Set the private variable A
  virtual inline MP_Param &setB(const arma::sp_mat &B) final {
    this->B = B;
    return *this;
  } ///< Set the private variable B
  virtual inline MP_Param &setc(const arma::vec &c) final {
    this->c = c;
    return *this;
  } ///< Set the private variable c
  virtual inline MP_Param &setb(const arma::vec &b) final {
    this->b = b;
    return *this;
  } ///< Set the private variable b

  virtual inline bool finalize() {
    this->size();
    return this->dataCheck();
  } ///< Finalize the MP_Param object.

  // Setters and advanced constructors
  virtual MP_Param &set(const arma::sp_mat &Q, const arma::sp_mat &C,
                        const arma::sp_mat &A, const arma::sp_mat &B,
                        const arma::vec &c,
                        const arma::vec &b); // Copy data into this
  virtual MP_Param &set(arma::sp_mat &&Q, arma::sp_mat &&C, arma::sp_mat &&A,
                        arma::sp_mat &&B, arma::vec &&c,
                        arma::vec &&b); // Move data into this
  virtual MP_Param &set(const QP_objective &obj, const QP_constraints &cons);

  virtual MP_Param &set(QP_objective &&obj, QP_constraints &&cons);

  virtual MP_Param &addDummy(unsigned int pars, unsigned int vars = 0,
                             int position = -1);

  void write(string filename, bool append = true) const;

  static bool dataCheck(const QP_objective &obj, const QP_constraints &cons,
                        bool checkObj = true, bool checkCons = true);
};

///@brief Class to handle parameterized quadratic programs(QP)
class QP_Param : public MP_Param
// Shape of C is Ny\times Nx
/**
 * Represents a Parameterized QP as \f[
 * \min_y \frac{1}{2}y^TQy + c^Ty + (Cx)^T y
 * \f]
 * Subject to
 * \f{eqnarray}{
 * Ax + By &\leq& b \\
 * y &\geq& 0
 * \f}
 */
{
private:
  // Gurobi environment and model
  GRBEnv *env;
  GRBModel QuadModel;
  bool made_yQy;

  int make_yQy();

public: // Constructors
  /// Initialize only the size. Everything else is empty (can be updated later)
  QP_Param(GRBEnv *env = nullptr)
      : env{env}, QuadModel{(*env)}, made_yQy{false} {
    this->size();
  }

  /// Set data at construct time
  QP_Param(arma::sp_mat Q, arma::sp_mat C, arma::sp_mat A, arma::sp_mat B,
           arma::vec c, arma::vec b, GRBEnv *env = nullptr)
      : env{env}, QuadModel{(*env)}, made_yQy{false} {
    this->set(Q, C, A, B, c, b);
    this->size();
    if (!this->dataCheck())
      throw string("Error in QP_Param::QP_Param: Invalid data for constructor");
  }

  /// Copy constructor
  QP_Param(const QP_Param &Qu)
      : MP_Param(Qu), env{Qu.env}, QuadModel{Qu.QuadModel}, made_yQy{
                                                                Qu.made_yQy} {
    this->size();
  };

  // Override setters
  QP_Param &set(const arma::sp_mat &Q, const arma::sp_mat &C,
                const arma::sp_mat &A, const arma::sp_mat &B,
                const arma::vec &c,
                const arma::vec &b) final; // Copy data into this
  QP_Param &set(arma::sp_mat &&Q, arma::sp_mat &&C, arma::sp_mat &&A,
                arma::sp_mat &&B, arma::vec &&c,
                arma::vec &&b) final; // Move data into this
  QP_Param &set(const QP_objective &obj, const QP_constraints &cons) final;

  QP_Param &set(QP_objective &&obj, QP_constraints &&cons) final;

  bool operator==(const QP_Param &Q2) const;

  // Other methods
  unsigned int KKT(arma::sp_mat &M, arma::sp_mat &N, arma::vec &q) const;

  std::unique_ptr<GRBModel> solveFixed(arma::vec x);

  double computeObjective(const arma::vec &y, const arma::vec &x,
                          bool checkFeas = true, double tol = 1e-6) const;

  inline bool is_Playable(const QP_Param &P) const
  /// Checks if the current object can play a game with another Game::QP_Param
  /// object @p P.
  {
    bool b1, b2, b3;
    b1 = (this->Nx + this->Ny) == (P.getNx() + P.getNy());
    b2 = this->Nx >= P.getNy();
    b3 = this->Ny <= P.getNx();
    return b1 && b2 && b3;
  }

  QP_Param &addDummy(unsigned int pars, unsigned int vars = 0,
                     int position = -1) override;

  void write(string filename, bool append) const;

  void save(string filename, bool erase = true) const;

  long int load(string filename, long int pos = 0);
};

/**
 * @brief Class to model Nash-cournot games with each player playing a QP
 */
/**
 * Stores a vector of QPs with each player's optimization problem.
 * Potentially common (leader) constraints can be stored too.
 *
 * Helpful in rewriting the Nash-Cournot game as an LCP
 * Helpful in rewriting leader constraints after incorporating dual variables
 * etc
 * @warning This has public fields which if accessed and changed can cause
 * undefined behavior!
 */
class NashGame {
private:
  GRBEnv *env = nullptr;
  arma::sp_mat LeaderConstraints; ///< Upper level leader constraints LHS
  arma::vec LeaderConsRHS;        ///< Upper level leader constraints RHS
  unsigned int Nplayers;          ///< Number of players in the Nash Game
  std::vector<shared_ptr<QP_Param>> Players; ///< The QP that each player solves
  arma::sp_mat MarketClearing;               ///< Market clearing constraints
  arma::vec MCRHS; ///< RHS to the Market Clearing constraints

  /// @internal In the vector of variables of all players,
  /// which position does the variable corrresponding to this player starts.
  std::vector<unsigned int> primal_position;
  ///@internal In the vector of variables of all players,
  /// which position do the DUAL variable corrresponding to this player starts.
  std::vector<unsigned int> dual_position;
  /// @internal Manages the position of Market clearing constraints' duals
  unsigned int MC_dual_position;
  /// @internal Manages the position of where the leader's variables start
  unsigned int Leader_position;
  /// Number of leader variables.
  /// These many variables will not have a matching complementary equation.
  unsigned int n_LeadVar;

  void set_positions();

public: // Constructors
  NashGame(GRBEnv *e)
      : env{e} {}; ///< To be used only when NashGame is being loaded from a
                   ///< file.
  NashGame(GRBEnv *e, std::vector<shared_ptr<QP_Param>> Players,
           arma::sp_mat MC, arma::vec MCRHS, unsigned int n_LeadVar = 0,
           arma::sp_mat LeadA = {}, arma::vec LeadRHS = {});

  NashGame(GRBEnv *e, unsigned int Nplayers, unsigned int n_LeadVar = 0,
           arma::sp_mat LeadA = {}, arma::vec LeadRHS = {})
      : env{e}, LeaderConstraints{LeadA},
        LeaderConsRHS{LeadRHS}, Nplayers{Nplayers}, n_LeadVar{n_LeadVar} {
    Players.resize(this->Nplayers);
    primal_position.resize(this->Nplayers);
    dual_position.resize(this->Nplayers);
  }

  /// Destructors to `delete` the QP_Param objects that might have been used.
  ~NashGame(){};

  // Verbose declaration
  friend ostream &operator<<(ostream &os, const NashGame &N) {
    os << endl;
    os << "--------------------------------------------------------------------"
          "---"
       << endl;
    os << "Nash Game with " << N.Nplayers << " players" << endl;
    os << "--------------------------------------------------------------------"
          "---"
       << endl;
    os << "Number of primal variables:\t\t\t " << N.getNprimals() << endl;
    os << "Number of dual variables:\t\t\t " << N.getNduals() << endl;
    os << "Number of shadow price dual variables:\t\t " << N.getNshadow()
       << endl;
    os << "Number of leader variables:\t\t\t " << N.getNleaderVars() << endl;
    os << "--------------------------------------------------------------------"
          "---"
       << endl;
    return os;
  }

  /// Return the number of primal variables
  inline unsigned int getNprimals() const {
    return this->primal_position.back();
  }

  inline unsigned int getNshadow() const { return this->MCRHS.n_rows; }

  inline unsigned int getNleaderVars() const { return this->n_LeadVar; }

  inline unsigned int getNduals() const {
    return this->dual_position.back() - this->dual_position.front() + 0;
  }

  // Size of variables
  inline unsigned int getPrimalLoc(unsigned int i = 0) const {
    return primal_position.at(i);
  }

  inline unsigned int getMCdualLoc() const { return MC_dual_position; }

  inline unsigned int getLeaderLoc() const { return Leader_position; }

  inline unsigned int getDualLoc(unsigned int i = 0) const {
    return dual_position.at(i);
  }

  // Members
  const NashGame &FormulateLCP(arma::sp_mat &M, arma::vec &q, perps &Compl,
                               bool writeToFile = false,
                               string M_name = "dat/LCP.txt",
                               string q_name = "dat/q.txt") const;

  arma::sp_mat RewriteLeadCons() const;

  inline arma::vec getLeadRHS() const { return this->LeaderConsRHS; }

  inline arma::vec getMCLeadRHS() const {
    return arma::join_cols(arma::join_cols(this->LeaderConsRHS, this->MCRHS),
                           -this->MCRHS);
  }

  // Check solution and correctness
  unique_ptr<GRBModel> Respond(unsigned int player, const arma::vec &x,
                               bool fullvec = true) const;

  double RespondSol(arma::vec &sol, unsigned int player, const arma::vec &x,
                    bool fullvec = true) const;

  arma::vec ComputeQPObjvals(const arma::vec &x, bool checkFeas = false) const;

  bool isSolved(const arma::vec &sol, unsigned int &violPlayer,
                arma::vec &violSol, double tol = 1e-6) const;

  //  Modify NashGame members

  NashGame &addDummy(unsigned int par = 0, int position = -1);

  NashGame &addLeadCons(const arma::vec &a, double b);

  // Read/Write Nashgame functions

  void write(string filename, bool append = true, bool KKT = false) const;

  void save(string filename, bool erase = true) const;

  long int load(string filename, long int pos = 0);
};

// void MPEC(NashGame N, arma::sp_mat Q, QP_Param &P);
// ostream& operator<< (ostream& os, const QP_Param &Q);
// void MPEC(NashGame N, arma::sp_mat Q, QP_Param &P);
ostream &operator<<(ostream &os, const QP_Param &Q);

ostream &operator<<(ostream &ost, const perps &C);

void print(const perps &C);
} // namespace Game

// The EPEC stuff
namespace Game {
/// @brief Stores statistics for a (solved) EPEC instance
struct EPECStatistics {
  int status = {0}; ///< status: 1: nashEq found. 0:no nashEq found. 2:timeLimit
  int numVar = {-1};         ///< Number of variables in findNashEq model
  int numConstraints = {-1}; ///< Number of constraints in findNashEq model
  int numNonZero = {-1}; ///< Number of non-zero coefficients in the constraint
                         ///< matrix of findNashEq model
  std::vector<int> feasiblePolyhedra =
      {}; ///< Vector containing the number of non-void polyhedra, indexed by
          ///< leader (country)
  double wallClockTime = {-1.0};
};

class EPEC {
private:
  std::vector<unsigned int> SizesWithoutHull{};
  int algorithm = 1; ///< Stores the type of algorithm used by the EPEC. 0 is
                     ///< FullEnumeration, 1 is Inner Approximation

protected: // Datafields
  std::vector<shared_ptr<Game::NashGame>> countries_LL{};
  std::vector<unique_ptr<Game::LCP>> countries_LCP{};

  std::vector<arma::sp_mat>
      LeadConses{};                   ///< Stores each leader's constraint LHS
  std::vector<arma::vec> LeadRHSes{}; ///< Stores each leader's constraint RHS

  std::vector<shared_ptr<Game::QP_Param>>
      country_QP{}; ///< The QP corresponding to each player
  std::vector<shared_ptr<Game::QP_objective>>
      LeadObjec{}; ///< Objective of each leader
  std::vector<shared_ptr<Game::QP_objective>>
      LeadObjec_ConvexHull{}; ///< Objective of each leader, given the convex
                              ///< hull computation

  unique_ptr<Game::NashGame> nashgame; ///< The EPEC nash game
  unique_ptr<Game::LCP> lcp;           ///< The EPEC nash game written as an LCP
  unique_ptr<GRBModel>
      lcpmodel; ///< A Gurobi mode object of the LCP form of EPEC

  std::vector<unsigned int> LeaderLocations{}; ///< Location of each leader
  std::vector<const unsigned int *> LocStarts{};
  std::vector<const unsigned int *> LocEnds{};
  std::vector<unsigned int> convexHullVarAddn{};
  std::vector<unsigned int> convexHullVariables{};
  unsigned int n_MCVar{0};

  GRBEnv *env;
  bool finalized{false}, nashEq{false};

  unsigned int nCountr{0};
  unsigned int nVarinEPEC{0};

  EPECStatistics Stats{}; ///< Store run time information

  arma::vec sol_z, ///< Solution equation values
      sol_x;       ///< Solution variable values

public:                  // Datafields
  bool indicators{true}; ///< Controls the flag @p useIndicators in Game::LCP.
                         ///< Uses @p bigM if @p false.
  double timeLimit{
      -1}; ///< Controls the timelimit for solve in Game::EPEC::findNashEq

private:
  virtual void add_Dummy_Lead(
      const unsigned int i) final; ///< Add Dummy variables for the leaders
  virtual void make_country_QP(const unsigned int i) final;
  virtual void make_country_QP() final;
  virtual void
  computeLeaderLocations(const unsigned int addSpaceForMC = 0) final;

  void giveAllDevns(std::vector<arma::vec> &devns,
                    const arma::vec &guessSol) const;
  void addDeviatedPolyhedron(const std::vector<arma::vec> &devns) const;
  virtual void computeNashEq() final;

protected: // functions
  EPEC(GRBEnv *env)
      : env{env}, timeLimit{
                      -1} {}; ///< Can be instantiated by a derived class only!

  // virtual function to be implemented by the inheritor.
  virtual void make_obj_leader(const unsigned int i,
                               Game::QP_objective &QP_obj) = 0;

  // virtual function to be optionally implemented by the inheritor.
  virtual void prefinalize();
  virtual void postfinalize();
  virtual void
  updateLocs() = 0; // If any location tracking system is implemented, that can
                    // be called from in here.
  virtual void make_MC_cons(arma::sp_mat &MC, arma::vec &RHS) const {
    MC.zeros();
    RHS.zeros();
  };

public:                  // functions
  EPEC() = delete;       ///< No default constructor
  EPEC(EPEC &) = delete; ///< Abstract class - no copy constructor
  ~EPEC() {}             ///< Destructor to free data

  virtual void finalize() final;
  virtual void findNashEq() final;

  virtual void iterativeNash() final;

  unique_ptr<GRBModel> Respond(const unsigned int i, const arma::vec &x) const;
  double RespondSol(arma::vec &sol, unsigned int player,
                    const arma::vec &x) const;
  bool isSolved(unsigned int *countryNumber, arma::vec *ProfDevn,
                double tol = 1e-6) const;

  virtual const arma::vec getx() const final { return this->sol_x; }
  void reset() { this->sol_x.ones(); }
  virtual const arma::vec getz() const final { return this->sol_z; }
  ///@brief Get the EPECStatistics object for the current instance
  virtual const EPECStatistics getStatistics() const final {
    return this->Stats;
  }
  virtual void setAlgorithm(unsigned int algorithm) final {
    switch (algorithm) {
    case 0:
      this->algorithm = 0;
      break;
    case 1:
      this->algorithm = 1;
      break;
    default:
      this->algorithm = 0;
    }
  }
};
} // namespace Game

/* Example for QP_Param */
/**
 * @page QP_Param_Example Game::QP_Param Example
 * Consider the Following quadratic program.
 * @f[
 * \min_{y_1, y_2, y_3} (y_1 + y_2 - 2y_3)^2 + 2 x_1y_1 + 2 x_2y_1 + 3 x_1y_3 +
 y_1-y_2+y_3
 * @f]
 * Subject to
 * @f{align*}{
 * y_1, y_2, y_3 &\ge 0 \\
 * y_1 + y_2 + y_3 &\le 10 \\
 * -y_1 +y_2 -2y_3 &\le -1 + x_1 + x_2
 * @f}
 *
 * This data can be entered as follows. Assume there are lines
 * @code
 using namespace arma;
 * @endcode
 * somewhere earlier. Now, within some function, we have
 * @code
        unsigned int Nx = 2, Ny = 3, Ncons = 2;
        mat Qd(3, 3);					// Easier to create a
 dense matrix for this problem
        Qd << 1 << 1 << -2 << endr		// And convert that to a sparse
 matrix.
           << 1 << 1 << -2 << endr
           << -2 << -2 << 4 << endr;
        sp_mat Q = sp_mat(2 * Qd);		// The matrix for y^2 terms
        sp_mat C(3, 2);					// The matrix for x and
 y interaction C.zeros(); C(0, 0) = 2; C(0, 1) = 2; C(2, 0) = 3; vec c(3);
 // The vector for linear terms in y c << 1 << endr << -1 << endr << 1 << endr;
        sp_mat A(2, 2);					// Constraint matrix for
 x terms A.zeros(); A(1, 0) = -1; A(1, 1) = -1; mat Bd(2, 3); Bd << 1 << 1 << 1
 << endr << -1 << 1 << -2 << endr;
        sp_mat B = sp_mat(Bd);			// Constraint matrix for y terms
        vec b(2);
        b(0) = 10;
        b(1) = -1;
        GRBEnv env = GRBEnv();		// Now create Gurobi environment to
 handle any solving related calls.
 * @endcode
 * Now the required object can be constructed in multiple ways.
 * @code
        // Method 1: Make a call to the constructor
        Game::QP_Param q1(Q, C, A, B, c, b, &env);

        // Method 2: Using QP_Param::set member function
        Game::QP_Param q2(&env);
        q2.set(Q, C, A, B, c, b);

        // Method 3: Reading from a file. This requires that such an object is
 saved to a file at first q1.save("dat/q1dat.dat"); // Saving the file so it can
 be retrieved. Game::QP_Param q3(&env); q3.load("dat/q1dat.dat");

        // Checking they are the same
        assert(q1==q2);
        assert(q2==q3);
 * @endcode
 *
 * With @f$(x_1, x_2) = (-1, 0.5)@f$, problem is
 * @f[
 * \min_{y_1, y_2, y_3} (y_1 + y_2 - 2y_3)^2  -y_2 -2y_3
 * @f]
 * Subject to
 * @f{align*}{
 * y_1, y_2, y_3 &\ge 0\\
 * y_1 + y_2 + y_3 &\le 10\\
 * -y_1 +y_2 -2y_3 &\le -1.5
 * @f}
 *
 * But this computation need not be done manually by supplying the value of
 @f$x@f$ and solving using Gurobi.
 * @code
        vec x(2);			// Enter the value of x in an arma::vec
        x(0) = -1;
        x(1) = 0.5;

        auto FixedModel = q2.solveFixed(x);	// Uses Gurobi to solve the
 model, returns a unique_ptr to GRBModel
 * @endcode
 * @p FixedModel has the GRBModel object, and all operations native to GRBModel,
 like accessing the value of a variable, a dual *	multiplier, saving the
 problem to a .lp file or a .mps file etc. can be performed on the object. In
 particular, the solution can be compared with hand-calculated solution as shown
 below. *	@code arma::vec sol(3); sol << 0.5417 << endr << 5.9861 << endr
 << 3.4722; // Hardcoding the solution as calculated outside for (unsigned int i
 = 0; i < Ny; i++) assert(abs(sol(i)-
 FixedModel->getVar(i).get(GRB_DoubleAttr_X)) <= 0.01);
        cout<<FixedModel->get(GRB_DoubleAttr_ObjVal<<endl; // Will print -12.757
 *	@endcode
 *
 * In many cases, one might want to obtain the KKT conditions of a convex
 quadratic program and that can be obtained as below, using @p QP_Param::KKT.
 *
 * The function returns @p M, @p N and @p q, where the KKT conditions can be
 written as
 * @f[
 * 0 \leq y \perp Mx + Ny + q \geq 0
 * @f]
 *
 * @code
 sp_mat M, N;
 vec q;
 q1.KKT(M, N, q);
 M.print("M");
 N.print("N");
 q.print("q");
 * @endcode
 *
 * Now that you are aware of most of the functionalities of Game::QP_Param, let
 us switch to the next tutorial on @link NashGame_Example NashGame and LCP
 @endlink.
 *
 */

/* Example of NashGame */
/**
 * @page NashGame_Example Game::NashGame and LCP Example
 *
 * Before reading this page, please ensure you are aware of the functionalities
 described in @link QP_Param_Example Game::QP_Param tutorial @endlink before
 following this page.
 *
 * @b PLAYER @b 1:
 * @f[
 * 	\min_{q_1}: 10 q_1 + 0.1 q_1^2 - (100 - (q_1+q_2)) q_1 	= 1.1 q_1^2 - 90
 q_1 + q_1q_2
 * 	@f]
 * 	 s.t:
 * 	 @f[
 * 	 	q_1 >= 0
 * 	 	@f]
 *
 *@b  PLAYER @b 2:
 * @f[
 * 	\min_{q_2}: 5 q_2 + 0.2 q_2^2 - (100 - (q_1+q_2)) q_2 	= 1.2 q_2^2 - 95
 q_2 + q_2q_1
 * 	@f]
 * 	 s.t:
 * 	 @f[
 * 	 	q_2 >= 0
 * 	 	@f]
 *
 * The above problem corresponds to a <a
 href="https://en.wikipedia.org/wiki/Cournot_competition">Cournot
 Competition</a> where the demand curve is given by @f$ P = a-BQ @f$ where @p P
 is the market price and @p Q is the quantity in the market. The cost of
 production of both the producers are given by a convex quadratic function in
 the quantity they produce. The solution to the problem is to find a <a
 href="https://en.wikipedia.org/wiki/Nash_equilibrium"> Nash equilibrium </a>
 from which neither producer is incentivized to deviate.
 *
 * To handle this problem, first we create two objects of Game::QP_Param to
 model each player's optimization problem, as parameterized by the other.
 * @code
        arma::sp_mat Q(1, 1), A(0, 1), B(0, 1), C(1, 1);
        arma::vec b, c(1);
        b.set_size(0);

        Q(0, 0) = 2 * 1.1;
        C(0, 0) = 1;
        c(0) = -90;
        auto q1 = std::make_shared<Game::QP_Param>(Q, C, A, B, c, b, &env);

        Q(0, 0) = 2 * 1.2;
        c(0) = -95;
        auto q2 = std::make_shared<Game::QP_Param>(Q, C, A, B, c, b, &env);

        std::vector<shared_ptr<Game::QP_Param>> q{q1, q2}; // Making a vector
 shared_ptr to the individual players' problem
 * @endcode
 *
 * Next, since we do not have any Market clearing constraints, we set empty
 matrices for them. Note that, if the problem does not have market clearing
 constraints, still the matrices have to be input with zero rows and appropriate
 number of columns.
 * @code
        sp_mat MC(0, 2);
        vec MCRHS;
        MCRHS.set_size(0);
 * @endcode
 * Finally now, we can make the Game::NashGame object by invoking the
 constructor.
 *
 * @code
 * 		GRBEnv env;
        Game::NashGame Nash = Game::NashGame(&env, q, MC, MCRHS);
 * @endcode
 *
 * Using traditional means, one can write a linear complementarity problem (LCP)
 to solve the above problem. The LCP is given as follows.
 *
 * <b> EXPECTED LCP </b>
 * @f{eqnarray*}{
 * 0 \le q_1 \perp 2.2 q_1 + q_2 - 90 \geq 0\\
 * 0 \le q_2 \perp q_1 + 2.4 q_2 - 95 \geq 0
 * @f}
 *
 * To observe the LCP formulation of this NashGame, one can use
 Game::NashGame::FormulateLCP member function.
 * @code
 * 	arma::sp_mat M;
 * 	arma::vec q;
 * 	perps Compl;		// Stores the complementarity pairs
 relationships.
 * 	Nash.FormulateLCP(M, q, Compl);	// Compute the LCP
 *
 * M.print();
 * q.print(); *
 * @endcode
 *
 * Here @p M and @p q are such that the solution to the LCP @f$ 0 \le x \perp Mx
 + q \ge 0 @f$ solves the original NashGame. These matrices can be written to a
 file and solved externally now.
 *
 * Alternatively, one can pass it to the Game::LCP class, and solve it natively.
 To achieve this, one can pass the above matrices to the constructor of the
 Game::LCP class.
 * @code
        GRBEnv env = GRBEnv();
        Game::LCP lcp = Game::LCP(&env, M, q, 1, 0);
 * @endcode
 *
 * More concisely, the class Game::LCP offers a constructor with the NashGame
 itself as an argument. This way, one need not explicitly compute @p M, @p q
 etc., to create the Game::LCP object.
 * @code
        Game::LCP lcp2 = Game::LCP(&env, Nash);
 * @endcode
 *
 * Now the Game::LCP object can be solved. And indeed the solution helps obtain
 the Nash equilibrium of the original Nash game.
 * @code
        auto model = lcp.LCPasMIP();
        model.optimize();			// Alternatively, auto model =
 lcp.LCPasMIP(true); will already optimize and solve the model.
 * @endcode
 * As was the case with Game::QP_Param::solveFixed, the above function returns a
 unique_ptr to GRBModel. And all native operations to the GRBModel can be
 performed and the solution be obtained.
 *
 * The solution to this problem can be obtained as @f$q_1=28.271028@f$,
 @f$q_2=27.803728@f$. To indeed check that this solution is correct, one can
 create a solution vector and solve each player's Game::QP_Param and check that
 the solution indeed matches.
 * @code
                arma::vec Nashsol(2);
                Nashsol(0) = model->getVarByName("x_0").get(GRB_DoubleAttr_X);
 // This is 28.271028 Nashsol(1) =
 model->getVarByName("x_1").get(GRB_DoubleAttr_X); // This is 27.803728

                auto nashResp1 = Nash.Respond(0, Nashsol);
                auto nashResp2 = Nash.Respond(1, Nashsol);

                cout<<nashResp1->getVarByName("y_0").get(GRB_DoubleAttr_X)<<endl;
 // Should print 28.271028
                cout<<nashResp2->getVarByName("y_0").get(GRB_DoubleAttr_X)<<endl;
 // Should print 27.803728
 * @endcode
 * One can, thus check that the values match the solution values obtained
 earlier. If only does not want the individual GRBModel handles, but just want
 to confirm either that the problem is solved or to provide a player with
 profitable deviation, one can just use Game::NashGame::isSolved function as
 follows.
 * @code
                unsigned int temp1 ; arma::vec temp2;
                cout<<Nash.isSolved(Nashsol, temp1, temp2); // This should be
 true.
 * @endcode
 * If the Game::NashGame::isSolved function returns false, then @p temp1 and @p
 temp2 respectively contain the player with profitable deviation, and the more
 profitable strategy of the player.
 *
 * And note that, just like Game::QP_Param, Game::NashGame can also be saved to
 and loaded from an external file.
 * @code
        Nash.save("dat/Nash.dat"); //Saves the object
        Game::NashGame Nash2(&env);
        Nash2.load("dat/Nash.dat"); // Loads the object into memory.
 * @endcode
 * Now that you are aware of most of the functionalities of Game::NashGame, let
 us switch to the next tutorial on @link LCP_Example LCP @endlink.
 *
 */
