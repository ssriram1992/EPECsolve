#include "epectests.h"

// Getting Follower parameter
Models::FollPar FP_Rosso();
Models::FollPar FP_Bianco();
Models::FollPar FP_C3F1();
Models::FollPar OneGas();
Models::FollPar OneCoal();
Models::FollPar OneSolar();
arma::sp_mat TranspCost(unsigned int n);


Models::LeadAllPar LAP_LowDem(Models::FollPar followers, Models::LeadPar leader, std::string a="") ;

Models::LeadAllPar LAP_HiDem(Models::FollPar followers, Models::LeadPar leader, std::string a="") ;

struct countrySol {
  std::vector<double> foll_prod;
  std::vector<double> foll_tax;
  double export_;
  double import;
  double export_price;
};

struct testInst {
  Models::EPECInstance instance = {{}, {}};
  std::vector<countrySol> solution;
};

testInst CH_S_F0_CL_SC_F0() ;
testInst HardToEnum_1();
testInst HardToEnum_2();

std::vector <Game::EPECAlgorithmParams> allAlgo();
