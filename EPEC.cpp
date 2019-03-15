#include<iostream>
#include<exception>
#include"func.h"
#include<ctime>
#include<cstdlib>
#include<gurobi_c++.h>
#include<armadillo>
#include<iostream>

using namespace std;


/*
int LCPtest()
{
	GRBEnv env = GRBEnv();
	GRBModel* model=nullptr;
	arma::sp_mat M;		 arma::vec q;		 perps Compl;

	NashGame *Country = Models::createCountry(3, {3,2,1}, {1,20,5}, {100, 100, 100}, 80, 5, 10, 10, 0.5);
	
	bool Error{true};
	try
	{
		Country->FormulateLCP(M, q, Compl);
		Error = false;
	} 
	catch(const char* e) { cout<<e<<endl; }
	catch(string e) { cout<<"String: "<<e<<endl; }
	catch(exception &e) { cout<<"Exception: "<<e.what()<<endl; }
	catch(GRBException &e) {cout<<"GRBException: "<<e.getErrorCode()<<": "<<e.getMessage()<<endl;}
	if(Error) throw -1;
	M.print_dense("M"); 
	q.print("q");
	M.save("M.txt", arma::coord_ascii);
	q.save("q.txt", arma::arma_ascii);

	// game2LCPtest(M,q,Compl);
	LCP MyNashGame = LCP(&env, M, q, Compl);
	arma::sp_mat Aa ;//= new arma::sp_mat();
	arma::vec b ;//= new arma::vec();
	cout<<Compl<<endl;
	try
	{
		auto A = MyNashGame.BranchAndPrune();
		cout<<A->size()<<endl;
		for(auto v:*A)
		{
			for(auto u:*v) cout<<u<<"\t";
			cout<<endl;
		} 
		MyNashGame.print();
		cout<<"************************************"<<endl;
		MyNashGame.ConvexHull(&Aa, &b);
		cout<<"************************************"<<endl;
	}
	catch(const char* e) { cout<<e<<endl; }
	catch(string e) { cout<<"String: "<<e<<endl; }
	catch(exception &e) { cout<<"Exception: "<<e.what()<<endl; }
	catch(GRBException &e) {cout<<"GRBException: "<<e.getErrorCode()<<": "<<e.getMessage()<<endl;}
	delete model;
	delete Country;
	cout<<"Writing to files..."<<endl;
	b.save("b.txt", arma::arma_ascii);
	Aa.save("A.txt", arma::coord_ascii);
	cout<<Aa.n_rows<<", "<<Aa.n_cols<<endl;
	return 0;
}
*/



int main()
{
	Models::DemPar P;
	Models::FollPar *FP = new Models::FollPar();
	Models::LeadPar L;
	Models::LeadAllPar LA(3);
	FP->capacities = {-1, 15, 10};
	FP->costs_lin = {30, 40, 50};
	FP->costs_quad = {60, 40, 0};
	cout<<P<<*FP<<L;
	cout<<"Ini Full"<<endl;
	cout<<LA;
	return 0;
}
