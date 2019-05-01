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
		epec.make_country_QP();
		epec.findNashEq(true);
		cout<<"--------------------------------------------------Printing Locations--------------------------------------------------\n";
		for(unsigned int i = 0; i<epec.nCountries; i++)
		{
			cout<<"********** Country number "<<i+1<<"\t\t"<<"**********\n";
			for(int j=0; j<9;j++)
			{
				auto v = static_cast<Models::LeaderVars>(j);
				cout<<v<<"\t\t\t"<<epec.getPosition(i, v)<<endl;
			}
			cout<<endl;
		}
		cout<<"--------------------------------------------------Printing Locations--------------------------------------------------\n";
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
	Models::FollPar FP, FP2, FP3, FP1;
	Models::LeadPar L (0.3,-1,-1,10);
	FP.capacities = {10, 15};
	FP.costs_lin = {30, 40};
	FP.costs_quad = {60, 40}; 
	FP.emission_costs = {0, 0}; 

	FP1.capacities = {1000};
	FP1.costs_lin = {1};
	FP1.costs_quad = {0};
	FP1.emission_costs = {0};


	FP2.capacities = {10, 10};
	FP2.costs_lin = {30, 50};
	FP2.costs_quad = {60, 40}; 
	FP2.emission_costs = {10, 0}; 

	FP3.capacities = {10, 15, 5};
	FP3.costs_lin = {30, 40, 5};
	FP3.costs_quad = {60, 40, 10}; 
	FP3.emission_costs = {0, 0, 10}; 



	Models::LeadAllPar LA(2, "A", FP);
	Models::LeadAllPar LA1(1, "A1", FP1);
	Models::LeadAllPar LA1b(1, "A2", FP1);
	Models::LeadAllPar LA2(2, "B", FP2, {60,1});
	Models::LeadAllPar LA3(3, "C", FP3, {90,1});

	// Two followers Leader with price cap
	Models::LeadAllPar LA_pc1(1, "LA_pc1", FP1, {40,2}, L );
	Models::LeadAllPar LA_pc2(1, "LA_pc2", FP1, {60,3}, L );

	// cout<<LA<<LA2;
	// cout<<LA.FollowerParam.capacities.size()<<" "<<LA.FollowerParam.costs_lin.size()<<" "<<LA.FollowerParam.costs_quad.size()<<endl;
	arma::mat TrCo(2,2); 
	TrCo << 0 << 1<< arma::endr << 2 <<0;
	cout<<LA1<<LA2<<endl;
	LCPtest(LA_pc1, LA_pc2, arma::sp_mat(TrCo));


	return 0;
}

