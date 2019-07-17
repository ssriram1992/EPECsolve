#include<iostream>
#include<memory>
#include<exception>
#include"models.h"
#include<ctime>
#include<cstdlib>
#include<gurobi_c++.h>
#include<armadillo>
#include<iostream>
#include<iomanip>

using namespace std;


int LCPtest(Models::LeadAllPar LA, Models::LeadAllPar LA2, arma::sp_mat TrCo) {
    GRBEnv env = GRBEnv();
    // GRBModel* model=nullptr;
    arma::sp_mat M;
    arma::vec q;
    perps Compl;
    // Game::LCP *MyNashGame = nullptr;
    arma::sp_mat Aa;
    arma::vec b;
    Models::EPEC epec(&env);
    try {
        epec.addCountry(LA, 0).addCountry(LA2, 0).addTranspCosts(TrCo).finalize();
        epec.make_country_QP();
        try { epec.testQP(0); } catch (...) {}
        try { epec.testQP(1); } catch (...) {}
        epec.testCountry(1);
        epec.testCountry(0);
        epec.findNashEq(true);
        cout
                << "--------------------------------------------------Printing Locations--------------------------------------------------\n";
        for (unsigned int i = 0; i < epec.nCountries; i++) {
            cout << "********** Country number " << i + 1 << "\t\t" << "**********\n";
            for (int j = 0; j < 9; j++) {
                auto v = static_cast<Models::LeaderVars>(j);
                cout << Models::prn::label << std::setfill('.') << v << Models::prn::val << std::setfill('.')
                     << epec.getPosition(i, v) << endl;
            }
            cout << endl;
        }
        cout
                << "--------------------------------------------------Printing Locations--------------------------------------------------\n";
    }
    catch (const char *e) {
        cerr << e << endl;
        throw;
    }
    catch (string e) {
        cerr << "String: " << e << endl;
        throw;
    }
    catch (exception &e) {
        cerr << "Exception: " << e.what() << endl;
        throw;
    }
    catch (GRBException &e) {
        cerr << "GRBException: " << e.getErrorCode() << ": " << e.getMessage() << endl;
        throw;
    }
    return 0;
}


int main() {
    Models::DemPar P;
    Models::FollPar FP, FP2, FP3, FP1, FP3a;

    Models::LeadPar L(0.4, -1, -1, -1);

    FP1.capacities = {1000};
    FP1.costs_lin = {10};
    FP1.costs_quad = {0.1};
    FP1.emission_costs = {0};

    FP.capacities = {10, 15};
    FP.costs_lin = {0, 4};
    FP.costs_quad = {0, 40};
    FP.emission_costs = {0, 0};

    FP2.capacities = {10, 10};
    FP2.costs_lin = {30, 50};
    FP2.costs_quad = {20, 40};
    FP2.emission_costs = {10, 0};

    FP3.capacities = {10, 15, 50};
    FP3.costs_lin = {30, 32, 5};
    FP3.costs_quad = {60, 40, 10};
    FP3.emission_costs = {0, 1, 10};
    FP3.names = {"Solar producer", "Gas producer", "Coal producer"};

    FP3a.capacities = {10, 15, 50};
    FP3a.costs_lin = {30, 32, 5};
    FP3a.costs_quad = {60, 40, 10};
    FP3a.emission_costs = {0, 1, 10};
    FP3a.names = {"Solar producer", "Gas producer", "Coal producer"};
/*
*/

    // Two followers Leader with price cap
    Models::LeadAllPar LA_pc1(3, "USA", FP3, {40, 0.10}, {0.4, -1, -1, -1});
    Models::LeadAllPar LA_pc2(3, "China", FP3, {60, 0.25}, {0.4, -1, -1, -1});

    // cout<<LA<<LA2;
    // cout<<LA.FollowerParam.capacities.size()<<" "<<LA.FollowerParam.costs_lin.size()<<" "<<LA.FollowerParam.costs_quad.size()<<endl;
    arma::mat TrCo(2, 2);
    TrCo << 0 << 0.5 << arma::endr << 0.5 << 0;
    LCPtest(LA_pc1, LA_pc2, arma::sp_mat(TrCo));


    return 0;
}
