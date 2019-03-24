#include<iostream>
#include<exception>
#include"func.h"
#include<ctime>
#include<cstdlib>
#include<gurobi_c++.h>
#include<armadillo>
#include<iostream>

using namespace std;



int main()
{
	GRBEnv env = GRBEnv();
	GRBModel* model=nullptr;
	arma::sp_mat M;		 arma::vec q;		 perps Compl;
	LCP *MyNashGame = nullptr;
	arma::sp_mat Aa; arma::vec b;
	bool Error{true};
	try
	{
		MyNashGame = Models::createCountry(env, 2, {3,1}, {1,5}, {100, 100}, 80, 5, 10, 10, 0.5);
		Error = false;
	} 
	catch(const char* e) { cout<<e<<endl; }
	catch(string e) { cout<<"String: "<<e<<endl; }
	catch(exception &e) { cout<<"Exception: "<<e.what()<<endl; }
	catch(GRBException &e) {cout<<"GRBException: "<<e.getErrorCode()<<": "<<e.getMessage()<<endl;}
	if(Error) throw -1;
	// game2LCPtest(M,q,Compl);
	try
	{
		auto A = MyNashGame->BranchAndPrune();
		cout<<A->size()<<endl;
		for(auto v:*A)
		{
			for(auto u:*v) cout<<u<<"\t";
			cout<<endl;
		} 
		MyNashGame->print();
		cout<<"************************************"<<endl;
		MyNashGame->ConvexHull(&Aa, &b);
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



