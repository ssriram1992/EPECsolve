#include "models.h"

Models::EPECInstance ChileArgentinaInstance() {
  Models::FollPar A, C;
  // steam groupped into geotherm
  std::vector<std::string> Names = {"Biomass", "Coal",  "Wind",   "Gas",
                                    "Geo",     "Hydro", "Diesel", "Solar"};
  arma::vec LinearCosts = {7.2, 59.2, 1, 90, 1, 1, 190, 5};
  arma::vec QuadraticCosts = {2.6e-06, 1.0e-06, 1.3e-06, 4.2e-06,
                              2.2e-05, 1.4e-08, 7.7e-05, 9.0e-07};
  arma::vec EmissionCosts = {195, 200, 0, 110, 0, 0, 180, 0};
  arma::vec ChileCapacities = {1958, 27084, 4528, 22589,
                               245,  21654, 2748, 4513};
  arma::vec ArgentinaCapacities = {0, 0, 2747, 93095, 0, 36587, 1584, 554};
  for (unsigned int i = 0; i < ChileCapacities.size(); ++i) {
    if (ChileCapacities.at(i) > 0) {
      C.capacities.push_back(ChileCapacities.at(i));
      C.names.push_back(Names.at(i));
      C.tax_caps.push_back(-1);
      C.costs_lin.push_back(LinearCosts.at(i));
      C.costs_quad.push_back(QuadraticCosts.at(i));
      C.emission_costs.push_back(EmissionCosts.at(i));
    }
    if (ArgentinaCapacities.at(i) > 0) {
      A.capacities.push_back(ArgentinaCapacities.at(i));
      A.names.push_back(Names.at(i));
      A.tax_caps.push_back(-1);
      A.costs_lin.push_back(LinearCosts.at(i));
      A.costs_quad.push_back(QuadraticCosts.at(i));
      A.emission_costs.push_back(EmissionCosts.at(i));
    }
  }
  Models::LeadAllPar Argentina(A.capacities.size(), "Argetina", A,
                               {132985, 0.05}, {-1, -1, 550});
  Models::LeadAllPar Chile(C.capacities.size(), "Chile", C, {69320, 0.03},
                           {-1, -1, 180});
  arma::sp_mat TrCo2(2, 2);
  TrCo2(1, 0) = 1;
  TrCo2(0, 1) = 1;
  Models::EPECInstance Instance2({Argentina, Chile}, TrCo2);
  Instance2.save("dat/ChileArgentina2_Q"); 
  return Instance2;

}

void solve(Models::EPECInstance instance)
{ 
    GRBEnv env;
    Models::EPEC epec(&env);
    const unsigned int nCountr = instance.Countries.size();
    for (unsigned int i = 0; i < nCountr; i++)
      epec.addCountry(instance.Countries.at(i)); 
        epec.addTranspCosts(instance.TransportationCosts);
    epec.finalize();
    epec.setAlgorithm(Game::EPECalgorithm::innerApproximation);
    epec.setAggressiveness(3);
    epec.setAddPolyMethod(Game::EPECAddPolyMethod::sequential); 
	std::cout << "Starting to solve...\n";
	epec.findNashEq();
	epec.writeSolution(2, "dat/ChileArgentina");
}

int main()
{
	solve(ChileArgentinaInstance());
	return 0;
}
