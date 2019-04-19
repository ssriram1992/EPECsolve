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
		cout<<"Here1\n";
		epec.make_country_QP(0);
		epec.make_country_QP(1);
	} 
	catch(const char* e) { cerr<<e<<endl;throw; }
	catch(string e) { cerr<<"String: "<<e<<endl;throw; }
	catch(exception &e) { cerr<<"Exception: "<<e.what()<<endl;throw; }
	catch(GRBException &e) {cerr<<"GRBException: "<<e.getErrorCode()<<": "<<e.getMessage()<<endl;throw;}
	return 0;
}



int main()
{
	Models::DemPar P;
	Models::FollPar FP, FP2;
	Models::LeadPar L;
	FP.capacities = {10, 15};
	FP.costs_lin = {30, 40};
	FP.costs_quad = {60, 40}; 
	FP.emission_costs = {0, 0}; 

	FP2.capacities = {10, 10};
	FP2.costs_lin = {30, 50};
	FP2.costs_quad = {60, 40}; 
	FP2.emission_costs = {10, 0}; 
	Models::LeadAllPar LA(2, "A", FP);
	Models::LeadAllPar LA2(2, "B", FP2, {60,1});
	// cout<<LA<<LA2;
	// cout<<LA.FollowerParam.capacities.size()<<" "<<LA.FollowerParam.costs_lin.size()<<" "<<LA.FollowerParam.costs_quad.size()<<endl;
	arma::mat TrCo(2,2); 
	TrCo << 0 << 1<< arma::endr << 2 <<0;
	LCPtest(LA, LA2, arma::sp_mat(TrCo));


	return 0;
}

