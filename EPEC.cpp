#include<iostream>
#include<memory>
#include<exception>
#include"func.h"
#include<ctime>
#include<cstdlib>
#include<gurobi_c++.h>
#include<armadillo>
#include<iostream>

using namespace std;


int LCPtest(Models::LeadAllPar LA, Models::LeadAllPar LA2, arma::sp_mat TrCo)
{
	GRBEnv env = GRBEnv();
	// GRBModel* model=nullptr;
	arma::sp_mat M;		 arma::vec q;		 perps Compl;
	// Game::LCP *MyNashGame = nullptr;
	arma::sp_mat Aa; arma::vec b;
	Models::EPEC epec(&env);
	try
	{
		epec.addCountry(LA, 0).addCountry(LA2, 0).addTranspCosts(TrCo).finalize();
	} 
	catch(const char* e) { cout<<e<<endl;throw; }
	catch(string e) { cout<<"String: "<<e<<endl;throw; }
	catch(exception &e) { cout<<"Exception: "<<e.what()<<endl;throw; }
	catch(GRBException &e) {cout<<"GRBException: "<<e.getErrorCode()<<": "<<e.getMessage()<<endl;throw;}
	return 0;
}



int main()
{
	Models::DemPar P;
	Models::FollPar FP;
	Models::LeadPar L;
	FP.capacities = {10, 15, 10};
	FP.costs_lin = {30, 40, 50};
	FP.costs_quad = {60, 40, 0}; 
	FP.emission_costs = {0, 0, 0}; 
	Models::LeadAllPar LA(3, "A", FP);
	Models::LeadAllPar LA2(3, "B", FP, {60,1});
	// cout<<LA<<LA2;
	cout<<LA.FollowerParam.capacities.size()<<" "<<LA.FollowerParam.costs_lin.size()<<" "<<LA.FollowerParam.costs_quad.size()<<endl;
	arma::mat TrCo(2,2); 
	TrCo << 0 << 1<< arma::endr << 2 <<0;
	LCPtest(LA, LA2, arma::sp_mat(TrCo));
	return 0;
}

