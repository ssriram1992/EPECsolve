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


int test_OneCountry(Models::LeadAllPar LA) {
    GRBEnv env = GRBEnv();
    arma::mat TrCo(1,1, arma::fill::ones);
    Models::EPEC epec(&env);
    try {
        epec.addCountry(LA, 0).addTranspCosts(arma::sp_mat(TrCo)).finalize();
        epec.make_country_QP();
        epec.testQP(0);
        epec.testCountry(0);
        cout
                << "--------------------------------------------------Printing Locations--------------------------------------------------\n";
            cout << "********** Country 0 **********\n";
            for (int j = 0; j < 9; j++) {
                auto v = static_cast<Models::LeaderVars>(j);
                cout << Models::prn::label << std::setfill('.') << v << Models::prn::val << std::setfill('.')
                     << epec.getPosition(0, v) << endl;
            }
            cout << endl;
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
    Models::FollPar FP;

    FP.capacities = {10};
    FP.costs_lin = {30};
    FP.costs_quad = {60};
    FP.emission_costs = {1};
    FP.names = {"Solar producer"};

    Models::LeadAllPar LA_pc1(1, "USA", FP, {10, 1.15}, {1, -1, -1, -1});
    test_OneCountry(LA_pc1);

    return 0;
}
