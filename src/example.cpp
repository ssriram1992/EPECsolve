#include "epecsolve.h"
#include <gurobi_c++.h>

class My_EPEC_Prob : public EPEC {
public:
  My_EPEC_Prob(GRBEnv *e) : EPEC(e) {
    nCountr = 0;
    n_MCVar = 0;
  }
  void addLeader(std::shared_ptr<Game::NashGame> N, const unsigned int i) {
    this->countries_LL.push_back(N);
    this->convexHullVariables.push_back(0);
    this->convexHullVarAddn.push_back(0);
    this->LeadConses.push_back(N->RewriteLeadCons());
    ends[i] = N->getNprimals();
    this->LocEnds.push_back(&ends[i]);
    this->nCountr++;
  }
  void printEnds()
  {
	  std::cout << ends[0] << " " << ends[1] << "\n";
  }

private:
  unsigned int ends[2];
  void updateLocs() override {
    ends[0] = this->convexHullVariables.at(0);
    ends[1] = this->convexHullVariables.at(1);
  }
  void make_obj_leader(const unsigned int i,
                       Game::QP_objective &QP_obj) override {
    QP_obj.Q.zeros(3, 3);
    QP_obj.C.zeros(3, 3);
    QP_obj.c.zeros(3);
    switch (i) {
    case 0: // uv_leader's objective
      QP_obj.C(1, 0) = 1;
      QP_obj.c(0) = 1;
      QP_obj.c(2) = -1;
      break;
    case 1: // xy_leader's objective
      QP_obj.C(1, 2) = 1;
      QP_obj.c(0) = -1;
      QP_obj.c(2) = 1;
      break;
    default:
      throw std::string("Invalid make_obj_leader");
    }
  }
};

std::shared_ptr<Game::NashGame> uv_leader(GRBEnv *env) {
  // 2 variable and 2 constraints
  arma::sp_mat Q(2, 2), C(2, 1), A(2, 1), B(2, 2);
  arma::vec c(2, arma::fill::zeros);
  arma::vec b(2, arma::fill::zeros);
  // Q remains as 0
  // C remains as 0
  // c
  c(0) = -1;
  c(1) = 1;
  // A
  A(0, 0) = -1;
  A(1, 0) = 1;
  // B
  B(0, 0) = 2;
  B(0, 1) = 1;
  B(1, 0) = 1;
  B(1, 1) = -2;
  auto foll = std::make_shared<Game::QP_Param>(Q, C, A, B, c, b, env);

  // Lower level Market clearing constraints - empty
  arma::sp_mat MC(0, 3);
  arma::vec MCRHS(0, arma::fill::zeros);

  arma::sp_mat LeadCons(2, 3);
  arma::vec LeadRHS(2);
  LeadCons(0, 0) = 1;
  LeadCons(0, 1) = 1;
  LeadCons(0, 2) = 1;
  LeadRHS(0) = 10;

  auto N = std::make_shared<Game::NashGame>(
      env, std::vector<std::shared_ptr<Game::QP_Param>>{foll}, MC, MCRHS, 1,
      LeadCons, LeadRHS);
  return N;
}

std::shared_ptr<Game::NashGame> xy_leader(GRBEnv *env) {
  // 2 variable and 2 constraints
  arma::sp_mat Q(2, 2), C(2, 1), A(2, 1), B(2, 2);
  arma::vec c(2, arma::fill::zeros);
  arma::vec b(2, arma::fill::zeros);
  // Q remains as 0
  // C remains as 0
  // c
  c(0) = -1;
  c(1) = 1;
  // A
  A(0, 0) = -1;
  A(1, 0) = 1;
  // B
  B(0, 0) = 1;
  B(0, 1) = -1;
  B(1, 0) = 1;
  B(1, 1) = -1;
  // b
  b(0) = -5;
  b(1) = 3;
  auto foll = std::make_shared<Game::QP_Param>(Q, C, A, B, c, b, env);

  // Lower level Market clearing constraints - empty
  arma::sp_mat MC(0, 3);
  arma::vec MCRHS(0, arma::fill::zeros);

  arma::sp_mat LeadCons(1, 3);
  arma::vec LeadRHS(1);
  LeadCons(0, 0) = 1;
  LeadCons(0, 1) = 1;
  LeadCons(0, 2) = 1;
  LeadRHS(0) = 10;
  // LeadCons(0, 0) = 0;
  // LeadCons(0, 1) = -1;
  // LeadCons(0, 2) = 1;
  // LeadRHS(1) = 0;

  auto N = std::make_shared<Game::NashGame>(
      env, std::vector<std::shared_ptr<Game::QP_Param>>{foll}, MC, MCRHS, 1,
      LeadCons, LeadRHS);
  return N;
}

int main() {
  GRBEnv env;
  My_EPEC_Prob epec(&env);
  // Adding uv_leader
  auto uv_lead = uv_leader(&env);
  epec.addLeader(uv_lead, 0);
  // Adding xy_leader
  auto xy_lead = xy_leader(&env);
  epec.addLeader(xy_lead, 1);
  // Finalize
  epec.finalize();
  // epec.setAlgorithm(Game::EPECalgorithm::innerApproximation);
  // Solve
  try{
  epec.findNashEq();
  } catch (std::string &s) { std:: cerr<<"Error: "<<s<<'\n';}
  auto M = epec.getLcpModel();
  M.write("dat/ex_model.lp");
  M.optimize();
  M.write("dat/ex_sol.sol");
  epec.printEnds();
  return 0;
}

/**
@page Epec_example Game::EPEC example
 
Consider the following problem: The first player is the u-v player, where the leader's decision variables are @f$u@f$ and the follower's decision variaables are @f$v@f$. The second player is the x-y player where the leader's and the follower's variables are @f$x@f$ and @f$y@f$ respectively.

The u-v player's optimization problem is given below
@f{align*}{
min_{u,v} 
\quad&:\quad
u + xv_1 - v_2 \\
\text{s.t.}\\
u,v \quad&\ge\quad 0\\
u+v_1+v_2 \quad&\leq\quad 10\\
v \quad&\in\quad
\arg \min _v
\left \{
-v_1+v_2 : 
v \ge 0; 
2v_1+v_2 \leq u;
v_1 -2v_2 \leq -u
\right \}
@f}
On a similar note, the optimization problem of the x-y player is given as follows.
@f{align*}{
\min_{x,y} \quad&:\quad
-x + v_2y_1 + y_2\\
\text{s.t.}\\
x, y \quad&\ge\quad 0\\
x + y_1 + y_2 \quad&\le\quad 10\\
y_1 - y_2 \quad&\ge\quad 0\\
y\quad&\in\quad\arg\min_y
\left\{
y_1 - y_2:
y_1 - y_2 \ge x-5; 
y_1 - y_2 \ge 3-x
\right\}
@f}

The solution obtained is @f$u = 2.5@f$, @f$(v_1, v_2) = (0, 2.5)@f$. Similarly for the other leader,
the solution obtained is @f$x = 0@f$, 
*/
