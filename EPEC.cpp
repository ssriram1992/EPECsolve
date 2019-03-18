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


int LCPtest(Models::LeadAllPar LA)
{
	GRBEnv env = GRBEnv();
	GRBModel* model=nullptr;
	arma::sp_mat M;		 arma::vec q;		 perps Compl;
	LCP *MyNashGame = nullptr;
	arma::sp_mat Aa; arma::vec b;
	bool Error{true};
	try
	{
		// MyNashGame = Models::createCountry(env, 2, {3,1}, {1,5}, {100, 100}, 80, 5, 10, 10, 0.5);
		MyNashGame = Models::createCountry(env, LA, 0);
		Error = false;
	} 
	catch(const char* e) { cout<<e<<endl; }
	catch(string e) { cout<<"String: "<<e<<endl; }
	catch(exception &e) { cout<<"Exception: "<<e.what()<<endl; }
	catch(GRBException &e) {cout<<"GRBException: "<<e.getErrorCode()<<": "<<e.getMessage()<<endl;}
	if(Error) throw -1;
	try
	{
		// Solving using LCPasQP
		unique_ptr<GRBModel> Model2 = MyNashGame->LCPasQP(true);
		arma::vec z,x;
		MyNashGame->extractSols(Model2.get(), z, x, true);
		cout<<"Sample solution"<<endl;
		x.print("x"); z.print("z");
		cout<<"*************************"<<endl;


		// Branch and prune solving and getting Conv hull
		auto A = MyNashGame->BranchAndPrune();
		cout<<A->size()<<endl;
		for(auto v:*A)
		{
			for(auto u:*v) cout<<u<<"\t";
			cout<<endl;
		} 
		MyNashGame->print();
		cout<<"************************************"<<endl;
		MyNashGame->ConvexHull(Aa, b);
		cout<<"************************************"<<endl;
	}
	catch(const char* e) { cout<<e<<endl; }
	catch(string e) { cout<<"String: "<<e<<endl; }
	catch(exception &e) { cout<<"Exception: "<<e.what()<<endl; }
	catch(GRBException &e) {cout<<"GRBException: "<<e.getErrorCode()<<": "<<e.getMessage()<<endl;}
	delete model;
	cout<<"Writing to files..."<<endl;
	b.save("b.txt", arma::arma_ascii);
	Aa.save("A.txt", arma::coord_ascii);
	cout<<Aa.n_rows<<", "<<Aa.n_cols<<endl;
	delete MyNashGame;
	return 0;
}



int main()
{
	Models::DemPar P;
	Models::FollPar *FP = new Models::FollPar();
	Models::LeadPar L;
	FP->capacities = {10, 15, 10};
	FP->costs_lin = {30, 40, 50};
	FP->costs_quad = {60, 40, 0};
	Models::LeadAllPar LA(3, *FP);
	cout<<LA;
	cout<<LA.FollowerParam.capacities.size()<<" "<<LA.FollowerParam.costs_lin.size()<<" "<<LA.FollowerParam.costs_quad.size()<<endl;
	LCPtest(LA);
	return 0;
}
