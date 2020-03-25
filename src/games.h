#pragma once
/**
 * @file src/games.h For Game theory related algorithms
 */
#include "epecsolve.h"
#include <armadillo>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>
#include <set>
#include <string>

using namespace Game;

template <class T>
std::ostream &operator<<(std::ostream &ost, std::vector<T> v) {
  for (auto elem : v)
    ost << elem << " ";
  ost << '\n';
  return ost;
}

template <class T, class S>
std::ostream &operator<<(std::ostream &ost, std::pair<T, S> p) {
  ost << "<" << p.first << ", " << p.second << ">";
  return ost;
}

namespace Game {
class polyLCP; // Forward declaration

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
bool isZero(arma::mat M, double tol = 1e-6) noexcept;

bool isZero(arma::sp_mat M, double tol = 1e-6) noexcept;

// bool isZero(arma::vec M, double tol = 1e-6);
///@brief struct to handle the objective params of MP_Param/QP_Param
///@details Refer QP_Param class for what Q, C and c mean.
typedef struct QP_objective {
  arma::sp_mat Q;
  arma::sp_mat C;
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
  virtual inline bool finalize() {
    this->size();
    return this->dataCheck();
  } ///< Finalize the MP_Param object.

public:
  // Default constructors
  MP_Param() = default;

  MP_Param(const MP_Param &M) = default;
  void bound(double bigM, unsigned int primals);

  // Getters and setters
  arma::sp_mat getQ() const {
    return this->Q;
  } ///< Read-only access to the private variable Q
  arma::sp_mat getC() const {
    return this->C;
  } ///< Read-only access to the private variable C
  arma::sp_mat getA() const {
    return this->A;
  } ///< Read-only access to the private variable A
  arma::sp_mat getB() const {
    return this->B;
  } ///< Read-only access to the private variable B
  arma::vec getc() const {
    return this->c;
  } ///< Read-only access to the private variable c
  arma::vec getb() const {
    return this->b;
  } ///< Read-only access to the private variable b
  unsigned int getNx() const {
    return this->Nx;
  } ///< Read-only access to the private variable Nx
  unsigned int getNy() const {
    return this->Ny;
  } ///< Read-only access to the private variable Ny

  MP_Param &setQ(const arma::sp_mat &Q) {
    this->Q = Q;
    return *this;
  } ///< Set the private variable Q
  MP_Param &setC(const arma::sp_mat &C) {
    this->C = C;
    return *this;
  } ///< Set the private variable C
  MP_Param &setA(const arma::sp_mat &A) {
    this->A = A;
    return *this;
  } ///< Set the private variable A
  MP_Param &setB(const arma::sp_mat &B) {
    this->B = B;
    return *this;
  } ///< Set the private variable B
  MP_Param &setc(const arma::vec &c) {
    this->c = c;
    return *this;
  } ///< Set the private variable c
  MP_Param &setb(const arma::vec &b) {
    this->b = b;
    return *this;
  } ///< Set the private variable b

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

  void write(std::string filename, bool append = true) const;

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
      throw std::string(
          "Error in QP_Param::QP_Param: Invalid data for constructor");
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

  /// Computes the objective value, given a vector @p y and
  /// a parameterizing vector @p x
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
  /// @brief  Writes a given parameterized Mathematical program to a set of
  /// files.
  void write(std::string filename, bool append) const;
  /// @brief Saves the @p Game::QP_Param object in a loadable file.
  void save(std::string filename, bool erase = true) const;
  /// @brief Loads the @p Game::QP_Param object stored in a file.
  long int load(std::string filename, long int pos = 0);
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
  std::vector<std::shared_ptr<QP_Param>>
      Players;                 ///< The QP that each player solves
  arma::sp_mat MarketClearing; ///< Market clearing constraints
  arma::vec MCRHS;             ///< RHS to the Market Clearing constraints

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
        /// To be used only when NashGame is being loaded from a file.
  explicit NashGame(GRBEnv *e) noexcept : env{e} {};
  /// Constructing a NashGame from a set of Game::QP_Param, Market clearing
  /// constraints
  explicit NashGame(GRBEnv *e, std::vector<std::shared_ptr<QP_Param>> Players,
                    arma::sp_mat MC, arma::vec MCRHS,
                    unsigned int n_LeadVar = 0, arma::sp_mat LeadA = {},
                    arma::vec LeadRHS = {});
  // Copy constructor
  NashGame(const NashGame &N);
  ~NashGame(){};

  // Verbose declaration
  friend std::ostream &operator<<(std::ostream &os, const NashGame &N) {
    os << '\n';
    os << "--------------------------------------------------------------------"
          "---"
       << '\n';
    os << "Nash Game with " << N.Nplayers << " players" << '\n';
    os << "--------------------------------------------------------------------"
          "---"
       << '\n';
    os << "Number of primal variables:\t\t\t " << N.getNprimals() << '\n';
    os << "Number of dual variables:\t\t\t " << N.getNduals() << '\n';
    os << "Number of shadow price dual variables:\t\t " << N.getNshadow()
       << '\n';
    os << "Number of leader variables:\t\t\t " << N.getNleaderVars() << '\n';
    os << "--------------------------------------------------------------------"
          "---"
       << '\n';
    return os;
  }

  /// @brief Return the number of primal variables.
  inline unsigned int getNprimals() const {
    /***
     * Number of primal variables is the sum of the "y" variables present in
     * each player's Game::QP_Param
     */
    return this->primal_position.back();
  }
  /// @brief Gets the number of Market clearing Shadow prices
  /**
   * Number of shadow price variables is equal to the number of Market clearing
   * constraints.
   */
  inline unsigned int getNshadow() const { return this->MCRHS.n_rows; }
  /// @brief Gets the number of leader variables
  /**
   * Leader variables are variables which do not have a complementarity relation
   * with any equation.
   */
  inline unsigned int getNleaderVars() const { return this->n_LeadVar; }
  /// @brief Gets the number of dual variables in the problem
  inline unsigned int getNduals() const {
    /**
     * This is the count of number of dual variables and that is indeed the sum
     * of the number dual variables each player has. And the number of dual
     * variables for any player is equal to the number of linear constraints
     * they have which is given by the number of rows in the player's
     * Game::QP_Param::A
     */
    return this->dual_position.back() - this->dual_position.front() + 0;
  }

  // Position of variables
  /// Gets the position of the primal variable of i th player
  inline unsigned int getPrimalLoc(unsigned int i = 0) const {
    return primal_position.at(i);
  }
  /// Gets the positin where the Market-clearing dual variables start
  inline unsigned int getMCdualLoc() const { return MC_dual_position; }
  /// Gets the positin where the Leader  variables start
  inline unsigned int getLeaderLoc() const { return Leader_position; }
  /// Gets the location where the dual variables start
  inline unsigned int getDualLoc(unsigned int i = 0) const {
    return dual_position.at(i);
  }

  // Members
  const NashGame &FormulateLCP(arma::sp_mat &M, arma::vec &q, perps &Compl,
                               bool writeToFile = false,
                               std::string M_name = "dat/LCP.txt",
                               std::string q_name = "dat/q.txt") const;
  arma::sp_mat RewriteLeadCons() const;
  inline arma::vec getLeadRHS() const { return this->LeaderConsRHS; }
  inline arma::vec getMCLeadRHS() const {
    return arma::join_cols(arma::join_cols(this->LeaderConsRHS, this->MCRHS),
                           -this->MCRHS);
  }

  // Check solution and correctness
  std::unique_ptr<GRBModel> Respond(unsigned int player, const arma::vec &x,
                                    bool fullvec = true) const;
  double RespondSol(arma::vec &sol, unsigned int player, const arma::vec &x,
                    bool fullvec = true) const;
  arma::vec ComputeQPObjvals(const arma::vec &x, bool checkFeas = false) const;
  bool isSolved(const arma::vec &sol, unsigned int &violPlayer,
                arma::vec &violSol, double tol = 1e-4) const;
  //  Modify NashGame members
  NashGame &addDummy(unsigned int par = 0, int position = -1);
  NashGame &addLeadCons(const arma::vec &a, double b);
  // Read/Write Nashgame functions
  void write(std::string filename, bool append = true, bool KKT = false) const;
  /// @brief Saves the @p Game::NashGame object in a loadable file.
  void save(std::string filename, bool erase = true) const;
  /// @brief Loads the @p Game::NashGame object stored in a file.
  long int load(std::string filename, long int pos = 0);
};

std::ostream &operator<<(std::ostream &os, const QP_Param &Q);

std::ostream &operator<<(std::ostream &ost, const perps &C);

void print(const perps &C) noexcept;
} // namespace Game

// The EPEC stuff
namespace Game {

enum class EPECsolveStatus {
  /**
   * Set of status in which the solution status of a Game::EPEC can be.
   */
  nashEqNotFound, ///< Instance proved to be infeasible.
  nashEqFound,    ///< Solution found for the instance.
  timeLimit,      ///< Time limit reached, nash equilibrium not found.
  numerical,      ///< Numerical issues
  unInitialized   ///< Not started to solve the problem.
};

enum class EPECalgorithm {
  fullEnumeration,    ///< Completely enumerate the set of polyhedra for all
                      ///< followers
  innerApproximation, ///< Perform increasingly better inner approximations in
  ///< iterations
  combinatorialPNE, ///< Perform a combinatorial-based search strategy to find a
                    ///< pure NE
  outerApproximation ///< Perform an increasingly improving outer approximation
                     ///< of the feasible region of each leader
};

///< Recovery strategies for obtaining a PNE with innerApproximation
enum class EPECRecoverStrategy {
  incrementalEnumeration, ///< Add random polyhedra in each iteration
  combinatorial ///< Triggers the combinatorialPNE with additional information
                ///< from innerApproximation
};

/// @brief Stores the configuration for EPEC algorithms
struct EPECAlgorithmParams {
  Game::EPECalgorithm algorithm = Game::EPECalgorithm::fullEnumeration;
  Game::EPECRecoverStrategy recoverStrategy =
      EPECRecoverStrategy::incrementalEnumeration;
  bool polyLCP{
      true}; ///< True if the algorithm extends the LCP to polyLCP. Namely, true if the algorithm uses the polyhedral class for the LCP
  Game::EPECAddPolyMethod addPolyMethod = Game::EPECAddPolyMethod::sequential;
  bool boundPrimals{false}; ///< If true, each QP param is bounded with an
                            ///< arbitrary large bigM constant
  double boundBigM{1e5}; ///< Bounding upper value if @p BoundPrimals is true.
  long int addPolyMethodSeed{
      -1}; ///< Random seed for the random selection of polyhedra. If -1, a
           ///< default computed value will be seeded.
  bool indicators{true}; ///< Controls the flag @p useIndicators in Game::LCP.
  ///< Uses @p bigM if @p false.
  double timeLimit{
      -1}; ///< Controls the timelimit for solve in Game::EPEC::findNashEq
  unsigned int threads{
      0}; ///< Controls the number of threads Gurobi exploits. Default 0 (auto)
  unsigned int aggressiveness{
      1}; ///< Controls the number of random polyhedra added at each iteration
  ///< in EPEC::iterativeNash
  bool pureNE{false}; ///< If true, the algorithm will tend to search for pure
  ///< NE. If none exists, it will return a MNE (if exists)
};

/// @brief Stores statistics for a (solved) EPEC instance
struct EPECStatistics {
  Game::EPECsolveStatus status = Game::EPECsolveStatus::unInitialized;
  int numVar = {-1};       ///< Number of variables in findNashEq model
  int numIteration = {-1}; ///< Number of iteration of the algorithm (not valid
                           ///< for fullEnumeration)
  int numConstraints = {-1}; ///< Number of constraints in findNashEq model
  int numNonZero = {-1}; ///< Number of non-zero coefficients in the constraint
  ///< matrix of findNashEq model
  int lostIntermediateEq = {0}; ///< Numer of times innerApproximation cannot
                                ///< add polyhedra basing on deviations
  bool numericalIssuesEncountered = {
      false}; ///< True if there have been some numerical issues during the
              ///< iteration of the innerApproximation
  std::vector<unsigned int> feasiblePolyhedra =
      {}; ///< Vector containing the number of non-void polyhedra, indexed by
          ///< leader (country)
  double wallClockTime = {0};
  bool pureNE{false}; ///< True if the equilibrium is a pure NE.
  EPECAlgorithmParams AlgorithmParam =
      {}; ///< Stores the configuration for the EPEC algorithm employed in the
          ///< instance.
};

///@brief Class to handle a Nash game between leaders of Stackelberg games
class EPEC {
private:
  std::vector<unsigned int> SizesWithoutHull{};
  Game::EPECalgorithm algorithm =
      Game::EPECalgorithm::fullEnumeration; ///< Stores the type of algorithm
  ///< used by the EPEC.
  std::unique_ptr<Game::LCP> lcp; ///< The EPEC nash game written as an LCP
  std::unique_ptr<GRBModel>
      lcpmodel; ///< A Gurobi mode object of the LCP form of EPEC
  std::unique_ptr<GRBModel>
      lcpmodel_base; ///< A Gurobi mode object of the LCP form of EPEC. If
                     ///< we are searching for a pure NE,
  ///< the LCP which is indifferent to pure or mixed NE is stored in this
  ///< object.
  unsigned int nVarinEPEC{0};
  unsigned int nCountr{0};

protected: // Datafields
  std::vector<std::shared_ptr<Game::NashGame>> countries_LL{};
  std::vector<std::shared_ptr<Game::LCP>> countries_LCP{};

  std::vector<std::shared_ptr<Game::QP_Param>>
      country_QP{}; ///< The QP corresponding to each player
  std::vector<std::shared_ptr<Game::QP_objective>>
      LeadObjec{}; ///< Objective of each leader
  std::vector<std::shared_ptr<Game::QP_objective>>
      LeadObjec_ConvexHull{}; ///< Objective of each leader, given the convex
                              ///< hull computation

  std::unique_ptr<Game::NashGame> nashgame; ///< The EPEC nash game

  std::vector<unsigned int> LeaderLocations{}; ///< Location of each leader
  /// Number of variables in the current player, including any number of convex
  /// hull variables at the current moment. The used, i.e., the inheritor of
  /// Game::EPEC has the responsibility to keep this correct by implementing an
  /// override of Game::EPEC::updateLocs.
  std::vector<const unsigned int *> LocEnds{};
  std::vector<unsigned int> convexHullVariables{};
  unsigned int n_MCVar{0};

  GRBEnv *env;
  bool finalized{false};
  bool nashEq{false};
  std::chrono::high_resolution_clock::time_point initTime;
  EPECStatistics Stats{};            ///< Store run time information
  arma::vec sol_z,                   ///< Solution equation values
      sol_x;                         ///< Solution variable values
  bool warmstart(const arma::vec x); ///< Warmstarts EPEC with a solution

private:
  void
  add_Dummy_Lead(const unsigned int i); ///< Add Dummy variables for the leaders
  void make_country_QP(const unsigned int i);
  void make_country_QP();
  void make_country_LCP();

  void make_pure_LCP(bool indicators = false);
  void computeLeaderLocations(const unsigned int addSpaceForMC = 0);

  void get_x_minus_i(const arma::vec &x, const unsigned int &i,
                     arma::vec &solOther) const;
  bool computeNashEq(bool pureNE = false, double localTimeLimit = -1.0,
                     bool check = false);

protected: // functions
  EPEC(GRBEnv *env)
      : env{env} {}; ///< Can be instantiated by a derived class only!

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
  bool hasLCP() const {
    if (this->lcp)
      return true;
    else
      return false;
  }

public: // functions
  // Friends algorithmic classes
  friend class Algorithms::PolyBase;
  friend class Algorithms::innerApproximation;
  friend class Algorithms::outerApproximation;
  friend class Algorithms::combinatorialPNE;
  friend class Algorithms::fullEnumeration;
  EPEC() = delete;       // No default constructor
  EPEC(EPEC &) = delete; // Abstract class - no copy constructor
  ~EPEC() {}             // Destructor to free data

  void finalize();
  void findNashEq();

  std::unique_ptr<GRBModel> Respond(const unsigned int i,
                                    const arma::vec &x) const;
  double RespondSol(arma::vec &sol, unsigned int player, const arma::vec &x,
                    const arma::vec &prevDev) const;
  bool isSolved(unsigned int *countryNumber, arma::vec *ProfDevn,
                double tol = 51e-4) const;

  bool isSolved(double tol = 51e-4) const;

  const arma::vec getx() const { return this->sol_x; }
  void reset() { this->sol_x.ones(); }
  const arma::vec getz() const { return this->sol_z; }
  ///@brief Get the EPECStatistics object for the current instance
  const EPECStatistics getStatistics() const { return this->Stats; }
  void setAlgorithm(Game::EPECalgorithm algorithm);
  Game::EPECalgorithm getAlgorithm() const {
    return this->Stats.AlgorithmParam.algorithm;
  }
  void setRecoverStrategy(Game::EPECRecoverStrategy strategy);
  Game::EPECRecoverStrategy getRecoverStrategy() const {
    return this->Stats.AlgorithmParam.recoverStrategy;
  }
  void setAggressiveness(unsigned int a) {
    this->Stats.AlgorithmParam.aggressiveness = a;
  }
  unsigned int getAggressiveness() const {
    return this->Stats.AlgorithmParam.aggressiveness;
  }
  void setNumThreads(unsigned int t) {
    this->Stats.AlgorithmParam.threads = t;
    this->env->set(GRB_IntParam_Threads, t);
  }
  unsigned int getNumThreads() const {
    return this->Stats.AlgorithmParam.threads;
  }
  void setAddPolyMethodSeed(unsigned int t) {
    this->Stats.AlgorithmParam.addPolyMethodSeed = t;
  }
  unsigned int getAddPolyMethodSeed() const {
    return this->Stats.AlgorithmParam.addPolyMethodSeed;
  }
  void setIndicators(bool val) { this->Stats.AlgorithmParam.indicators = val; }
  bool getIndicators() const { return this->Stats.AlgorithmParam.indicators; }
  void setPureNE(bool val) { this->Stats.AlgorithmParam.pureNE = val; }
  bool getPureNE() const { return this->Stats.AlgorithmParam.pureNE; }
  void setBoundPrimals(bool val) {
    this->Stats.AlgorithmParam.boundPrimals = val;
  }
  bool getBoundPrimals() const {
    return this->Stats.AlgorithmParam.boundPrimals;
  }
  void setBoundBigM(double val) { this->Stats.AlgorithmParam.boundBigM = val; }
  double getBoundBigM() const { return this->Stats.AlgorithmParam.boundBigM; }
  void setTimeLimit(double val) { this->Stats.AlgorithmParam.timeLimit = val; }
  double getTimeLimit() const { return this->Stats.AlgorithmParam.timeLimit; }
  void setAddPolyMethod(Game::EPECAddPolyMethod add) {
    this->Stats.AlgorithmParam.addPolyMethod = add;
  }
  Game::EPECAddPolyMethod getAddPolyMethod() const {
    return this->Stats.AlgorithmParam.addPolyMethod;
  }

  // Methods to get positions of variables
  // The below are all const functions which return an unsigned int.
  int getnVarinEPEC() const noexcept { return this->nVarinEPEC; }
  int getNcountries() const noexcept {
    return this->countries_LL.size();
  }
  unsigned int getPosition_LeadFoll(const unsigned int i,
                                    const unsigned int j) const;
  unsigned int getPosition_LeadLead(const unsigned int i,
                                    const unsigned int j) const;
  unsigned int getPosition_LeadFollPoly(const unsigned int i,
                                        const unsigned int j,
                                        const unsigned int k) const;
  unsigned int getPosition_LeadLeadPoly(const unsigned int i,
                                        const unsigned int j,
                                        const unsigned int k) const;
  unsigned int getNPoly_Lead(const unsigned int i) const;
  unsigned int getPosition_Probab(const unsigned int i,
                                  const unsigned int k) const;

  // The following obtain the variable values
  double getVal_LeadFoll(const unsigned int i, const unsigned int j) const;
  double getVal_LeadLead(const unsigned int i, const unsigned int j) const;
  double getVal_LeadFollPoly(const unsigned int i, const unsigned int j,
                             const unsigned int k,
                             const double tol = 1e-5) const;
  double getVal_LeadLeadPoly(const unsigned int i, const unsigned int j,
                             const unsigned int k,
                             const double tol = 1e-5) const;
  double getVal_Probab(const unsigned int i, const unsigned int k) const;

  // The following checks if the returned strategy leader is a pure strategy
  // for a leader or appropriately retrieve mixed-strategies
  bool isPureStrategy(const unsigned int i, const double tol = 1e-5) const;
  bool isPureStrategy(const double tol = 1e-5) const;
  std::vector<unsigned int> mixedStratPoly(const unsigned int i,
                                           const double tol = 1e-5) const;

  /// Get the Game::LCP object solved in the last iteration either to solve the
  /// problem or to prove non-existence of Nash equilibrium. Object is returned
  /// using constant reference.
  const LCP &getLcpDescr() const { return *this->lcp.get(); }
  /// Get the GRBModel solved in the last iteration to solve the problem or to
  /// prove non-existence of Nash equilibrium. Object is returned using constant
  /// reference.
  const GRBModel &getLcpModel() const { return *this->lcpmodel.get(); }
  /// Writes the GRBModel solved in the last iteration to solve the problem or
  /// to prove non-existence of Nash equilibrium to a file.
  void writeLcpModel(std::string filename) const {
    this->lcpmodel->write(filename);
  }
};
} // namespace Game

namespace std {
string to_string(const Game::EPECsolveStatus st);
string to_string(const Game::EPECalgorithm al);
string to_string(const Game::EPECRecoverStrategy st);
string to_string(const Game::EPECAlgorithmParams al);
string to_string(const Game::EPECAddPolyMethod add);
}; // namespace std

/* Example for QP_Param */

/* Example of NashGame */
