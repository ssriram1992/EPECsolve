#include "models.h"
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <string>

Models::EPECInstance ChileArgentinaInstance() {
  Models::FollPar Arg, Chi;
  // steam groupped into geotherm

  Chi.names = {"Coal", "Wind", "Gas", "Hydro", "Solar", "Diesel"};
  Arg.names = {"Wind", "Gas", "Hydro", "Solar", "Diesel"};

  // In million dollars per terawatt-hour of energy

  Models::FollPar Chi_Main{
      {0.2, 0.1, 0, 0},            // quad
      {30, 52, 0, 0},               // lin
      {27, 23, 22, 9},             // cap
      {200, 110, 0, 0},            // emm
      {50, 100, 20, 20},            // tax
      {"Coal", "Gas", "Hydro", "Non-conv renewable"}, // name
  };
  Models::FollPar Chi_Wind{{0}, {0}, {5}, {0}, {20}, {"Wind"}};
  Models::FollPar Chi_Solar{{0}, {0}, {5}, {0}, {20}, {"Solar"}};
  Models::FollPar Chi_Diesel{{200}, {0.1}, {3}, {180}, {200}, {"Diesel"}};

  Models::FollPar Arg_Main{
      {0, 0},           // quad
      {55, 0},          // lin
      {103, 37},        // cap
      {110, 0},         // emm
      {110, 20},        // tax
      {"Gas", "Hydro"}, // name
  };
  Models::FollPar Arg_Wind{{0}, {0}, {3}, {0}, {20}, {"Wind"}};
  Models::FollPar Arg_Solar{{0}, {0}, {3}, {0}, {20}, {"Solar"}};
  Models::FollPar Arg_Diesel{{115}, {0.1}, {2}, {180}, {115}, {"Diesel"}};

  Chi = Chi_Main;
  Arg = Arg_Main;

  Chi.tax_caps = Chi.costs_lin;
  Arg.tax_caps = Arg.costs_lin;

  Arg.tax_caps = {0.5, 0.5};
  Chi.tax_caps = {0.65, 0, 0, 0};

  // Chi.tax_caps = {Chi.costs_lin.at(0)/Chi.emission_costs.at(0),
  // Chi.costs_lin.at(1)/Chi.emission_costs.at(1),
  // Chi.costs_lin.at(2)/Chi.emission_costs.at(2)};
  // Arg.tax_caps = {Arg.costs_lin.at(0)/Arg.emission_costs.at(0),
  // Arg.costs_lin.at(1)/Arg.emission_costs.at(1)};
  /*
  Chi.costs_lin = { 50, 0, 100, 0, 0, 200 };
  Arg.costs_lin = {0, 105, 0, 0, 115};


  Chi.costs_quad = { 0.1, 0, 0.1, 0, 0, 0.1 };
  Arg.costs_quad = { 0, 0.1, 0, 0, 0.1 };

  Chi.emission_costs = { 200, 0, 110, 0, 0, 180 };
  Arg.emission_costs = { 0, 110, 0, 0, 180 };

  Chi.capacities = { 27, 5, 23, 22, 5, 3 };
  Arg.capacities = {3, 93, 37, 1, 2};

  Chi.tax_caps = {50, 20, 100, 20, 20, 200};
  Arg.tax_caps = {20, 105, 20, 20, 115};

  */
  Models::LeadAllPar Argentina(Arg.capacities.size(), "Argentina", Arg,
                               {800, 3.11}, {100, 100, -1, false, 2});
  Models::LeadAllPar Chile(Chi.capacities.size(), "Chile", Chi, {150, 1},
                           {100, 100, 90, false, 2});
  arma::sp_mat TrCo2(2, 2);
  TrCo2(1, 0) = 1;
  TrCo2(0, 1) = 1;
  Models::EPECInstance Instance2({Argentina, Chile}, TrCo2);
  Instance2.save("dat/ChileArgentina2_Q");
  return Instance2;
}

void solve(Models::EPECInstance instance) {
  GRBEnv env;
  Models::EPEC epec(&env);
  const unsigned int nCountr = instance.Countries.size();
  for (unsigned int i = 0; i < nCountr; i++)
    epec.addCountry(instance.Countries.at(i));
  epec.addTranspCosts(instance.TransportationCosts);
  epec.finalize();
  epec.setAlgorithm(Game::EPECalgorithm::fullEnumeration);
  epec.setAlgorithm(Game::EPECalgorithm::innerApproximation);
  epec.setAggressiveness(1);
  epec.setAddPolyMethod(Game::EPECAddPolyMethod::random);
  std::cout << "Starting to solve...\n";
  epec.setNumThreads(4);
  epec.findNashEq();
  epec.writeSolution(2, "dat/ChileArgentinaCarb");
}

int main() {
  boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                      boost::log::trivial::info);
  solve(ChileArgentinaInstance());
  return 0;
}
