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

int BilevelTest(Models::LeadAllPar LA) {
    GRBEnv env = GRBEnv();
    arma::sp_mat M;
    arma::vec q;
    perps Compl;
    arma::sp_mat Aa;
    arma::vec b;
    Models::EPEC epec(&env);
    try {
        arma::sp_mat TrCo(1, 1);
        TrCo(0, 0) = 0;
        epec.addCountry(LA).addTranspCosts(TrCo).finalize();
        epec.make_country_QP();
        //epec.testLCP(0);
        try { epec.testQP(0); } catch (...) { cerr << "Cannot test QP" << endl; }
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

int LCPtest(Models::LeadAllPar LA, Models::LeadAllPar LA2,
        // Models::LeadAllPar LA3,
            arma::sp_mat TrCo) {
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
        epec.addCountry(LA, 0).addCountry(LA2, 0)
                        // .addCountry(LA3,0)
                .addTranspCosts(TrCo).finalize();
        epec.make_country_QP();
        try { epec.testQP(0); } catch (...) {}
        try { epec.testQP(1); } catch (...) {}
        epec.testLCP(0);
        epec.testLCP(1);
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
    Models::FollPar FP, FP2, FP3, FP1, FP3a, FP2a;

    Models::LeadPar L(0.4, -1, -1, -1);

    FP1.capacities = {100};
    FP1.costs_lin = {10};
    FP1.costs_quad = {5};
    FP1.emission_costs = {6};
    FP1.names = {"US_follower"};

    FP.capacities = {100};
    FP.costs_lin = {4};
    FP.costs_quad = {0.25};
    FP.emission_costs = {10};
    FP.names = {"Eur_follower"};


    // Two followers Leader with price cap
    Models::LeadAllPar Europe(1, "Europe", FP, {80, 0.15}, {-1, -1, -1, -1});
    Models::LeadAllPar USA(1, "USA", FP1, {300, 0.05}, {-1, -1, -1, -1});
    // cout<<LA<<LA2;
    // cout<<LA.FollowerParam.capacities.size()<<" "<<LA.FollowerParam.costs_lin.size()<<" "<<LA.FollowerParam.costs_quad.size()<<endl;
    arma::mat TrCo(2, 2);
    TrCo << 0 << 1 << arma::endr << 1 << 0;
    //LCPtest(Europe, USA, static_cast<arma::sp_mat>(TrCo));
    BilevelTest(USA);


    return 0;
}