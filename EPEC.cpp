#include<iostream>
#include"func.h"
#include<ctime>
#include<cstdlib>
#include<gurobi_c++.h>
#include<armadillo>
#include<iostream>
#define VERBOSE true

using namespace std;

const int N_lead = 2;
const int L1F = 2;
const int L2F = 2;

array<double, L1F> CL1F{5,7};
array<double, L2F> CL2F{4,8};
array<double, L1F> CapL1F{500,700};
array<double, L2F> CapL2F{400,800};

const double L1F_alph = 1000;
const double L1F_beta = 1;

const double L2F_alph = 1200;
const double L2F_beta = 1.5;

int main()
{
	QP_Param *L1F1 = new QP_Param();
	QP_Param *L1F2 = new QP_Param();
	// For L1F1
	arma::sp_mat Q(1,1);
	Q(0,0) = L1F_beta;
	arma::vec c(1);
	c(0) = CL1F[0]-L1F_alph;
	arma::sp_mat C(1, 6);
	// Order Sould be: Other follower's variable, MC dual, q_imp^a, q_imp^B, t1, t2 
	C(0,0) = L1F_beta;
	C(0,4) = 1;
	C(0,2) = L1F_beta;
	// Constraints of L1F1
	arma::vec b(1);
	b(0)=CapL1F[0];
	arma::sp_mat A(1,6), B(1,1);
	B(0,0) = 1;
	L1F1->setMove(Q, C, A, B, c, b);
	// For L1F2
	Q.zeros(); c.zeros(); C.zeros(); A.zeros(); B.zeros(); b.zeros();
	Q(0,0) = L1F_beta;
	c(0) = CL1F[1]-L1F_alph;
	C(0,0) = L1F_beta;
	C(0,5) = 1;
	C(0,3) = L1F_beta;
	// Constraints of L1F1
	b(0)=CapL1F[1];
	B(0,0) = 1;
	L1F2->setMove(Q, C, A, B, c, b);

	// Market clearing constraints
	arma::sp_mat MC(1,7);
	MC(0,3)=1;MC(0,4)=-1;
	arma::vec MCRHS(1);
	MCRHS.zeros();

	// Nash Game
	//
	vector<QP_Param*> L1 {L1F1, L1F2};
	
	NashGame *MyGame = new NashGame(L1, MC, MCRHS, 4);
	arma::sp_mat M;
	arma::vec q;
	MyGame->FormulateLCP(M,q);
	M.print_dense("M"); 
	q.print();
	M.save("M.txt", arma::coord_ascii);
	q.save("q.txt", arma::arma_ascii);
	return 0;
}

int main2()
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
	Anew.save("Anew.txt",arma::coord_ascii);
	// A.print("A");
	// b.print("b");
	// 0 \leq x \perp Mx + Ny + q \geq 0 constraints in the MPEC.
	return 0;
}
