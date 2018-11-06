#include<iostream>
#include"func.h"
#include<ctime>
#include<cstdlib>
#include<gurobi_c++.h>
#include<armadillo>
#include<iostream>
#define VERBOSE 0

using namespace std;

const int N_lead = 2;
const int L1F = 2;
const int L2F = 2;

array<double, L1F> CL1F{5,7};
array<double, L2F> CL2F{5,7};

const double L1F_alph = 1000;
const double L1F_beta = 1;


int main()
{
	vector<arma::sp_mat> Ai{};
	vector<arma::vec> bi{};

	// for (int i=0;i<5;i++)
	// {
		// Ai.push_back(static_cast<arma::sp_mat>(arma::randi<arma::mat>(10+ i*i - i, 50, arma::distr_param(1,10))));
		// bi.push_back(arma::randi<arma::vec>(10 + i*i -i, arma::distr_param(1,i+2)));
	// }
	arma::sp_mat t1(3,2);
	arma::vec t2(3);
	t1(0,0)=-1; t1(0,1)=0; t1(1,0)=0; t1(1,1)=-1; t1(2,0)=-1; t1(2,1)=-1;
	t2(0)=0; t2(1)=4; t2(2)=1;
	Ai.push_back(t1);
	Ai.push_back(t1);
	Ai.push_back(t1);
	bi.push_back(t2);
	bi.push_back(t2);
	bi.push_back(t2);
	
	arma::sp_mat A;
	arma::vec b;
	// arma::sp_mat A(3,2);
	// arma::vec b(3);
	// A(0,0)=-1; A(0,1)=0; A(1,0)=0; A(1,1)=-1; A(2,0)=-1; A(2,1)=-1;
	// b(0)=0; b(1)=0; b(2)=5;
	// A.impl_raw_print_dense("A"); b.print();


	PolyUnion(Ai, bi, A, b);
	cout<<A.n_nonzero<<" non zeros in A which is "<<(float)A.n_nonzero/(A.n_cols*A.n_rows)*100<<"\% density"<<endl;
	cout<<"A's dimension is ("<<A.n_rows<<", "<<A.n_cols<<")"<<endl;

	b.print();
	arma::sp_mat Anew;
	arma::vec bnew;
	
	auto temp = makeCompactPolyhedron(A, b, Anew, bnew);
	cout<<endl<<endl;
	for(auto i:temp)
		cout<<i<<"\t";
	cout<<endl;
	Anew.impl_print_dense("A");
	bnew.print("b");
	// A.print("A");
	// b.print("b");
	// 0 \leq x \perp Mx + Ny + q \geq 0 constraints in the MPEC.
}
