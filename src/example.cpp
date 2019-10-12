#include "epecsolve.h"
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <gurobi_c++.h>

class My_EPEC_Prob : public EPEC {
public:
  My_EPEC_Prob(GRBEnv *e) : EPEC(e) {}
  void addLeader(std::shared_ptr<Game::NashGame> N, const unsigned int i) {
    this->countries_LL.push_back(N);
    ends[i] = N->getNprimals() + N->getNleaderVars();
    this->LocEnds.push_back(&ends[i]);
  }

private:
  unsigned int ends[2];
  void updateLocs() override {
    ends[0] = this->convexHullVariables.at(0) + 3;
    ends[1] = this->convexHullVariables.at(1) + 3;
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
      QP_obj.c(0) = 1;
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

  arma::sp_mat LeadCons(1, 3);
  arma::vec LeadRHS(1);
  LeadCons(0, 0) = 1;
  LeadCons(0, 1) = 1;
  LeadCons(0, 2) = 1;
  LeadRHS(0) = 5;

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
  c(0) = 1;
  c(1) = -1;
  // A
  A(0, 0) = 1;
  A(1, 0) = -1;
  // B
  B(0, 0) = -1;
  B(0, 1) = 1;
  B(1, 0) = -1;
  B(1, 1) = 1;
  // b
  b(0) = 5;
  b(1) = -3;
  auto foll = std::make_shared<Game::QP_Param>(Q, C, A, B, c, b, env);

  // Lower level Market clearing constraints - empty
  arma::sp_mat MC(0, 3);
  arma::vec MCRHS(0, arma::fill::zeros);

  arma::sp_mat LeadCons(2, 3);
  arma::vec LeadRHS(2);
  LeadCons(0, 0) = 1;
  LeadCons(0, 1) = 1;
  LeadCons(0, 2) = 1;
  LeadRHS(0) = 7;
  // Comment the following four lines for another example ;)
  LeadCons(1, 0) = -1;
  LeadCons(1, 1) = 1;
  LeadCons(1, 2) = 0;
  LeadRHS(1) = 0;

  auto N = std::make_shared<Game::NashGame>(
      env, std::vector<std::shared_ptr<Game::QP_Param>>{foll}, MC, MCRHS, 1,
      LeadCons, LeadRHS);
  return N;
}

int main() {
  GRBEnv env;
  boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                      boost::log::trivial::warning);
  My_EPEC_Prob epec(&env);
  // Adding uv_leader
  auto uv_lead = uv_leader(&env);
  epec.addLeader(uv_lead, 0);
  // Adding xy_leader
  auto xy_lead = xy_leader(&env);
  epec.addLeader(xy_lead, 1);
  // Finalize
  epec.finalize();
  epec.setAlgorithm(Game::EPECalgorithm::innerApproximation);
  // Solve
  try {
    epec.findNashEq();
  } catch (std::string &s) {
    std::cerr << "Error caught: " << s << '\n';
    throw;
  }
  // auto M = epec.getLcpModel();
  // M.write("dat/ex_model.lp");
  // M.optimize();
  // M.write("dat/ex_sol.sol");

  std::cout << "\nUV LEADER\n";
  std::cout << "u: " << epec.getVal_LeadLead(0, 0) << '\n';
  std::cout << "v_1: " << epec.getVal_LeadFoll(0, 0) << '\n';
  std::cout << "v_2: " << epec.getVal_LeadFoll(0, 1) << '\n';
  auto uv_strats = epec.mixedStratPoly(0);
  std::for_each(std::begin(uv_strats), std::end(uv_strats),
                [&epec](const unsigned int i) {
                  std::cout << "With probability  " << epec.getVal_Probab(0, i)
                            << '\n';
                  std::cout << "(" << epec.getVal_LeadLeadPoly(0, 0, i) << ", "
                            << epec.getVal_LeadFollPoly(0, 0, i) << ", "
                            << epec.getVal_LeadFollPoly(0, 1, i) << ")\n";
                });
  std::cout << '\n';
  std::cout << "\nXY LEADER\n";
  std::cout << "x: " << epec.getVal_LeadLead(1, 0) << '\n';
  std::cout << "y_1: " << epec.getVal_LeadFoll(1, 0) << '\n';
  std::cout << "y_2: " << epec.getVal_LeadFoll(1, 1) << '\n';
  auto xy_strats = epec.mixedStratPoly(1);
  std::for_each(std::begin(xy_strats), std::end(xy_strats),
                [&epec](const unsigned int i) {
                  std::cout << "With probability  " << epec.getVal_Probab(1, i)
                            << '\n';
                  std::cout << "(" << epec.getVal_LeadLeadPoly(1, 0, i) << ", "
                            << epec.getVal_LeadFollPoly(1, 0, i) << ", "
                            << epec.getVal_LeadFollPoly(1, 1, i) << ")\n";
                });
  std::cout << '\n';
  return 0;
}
