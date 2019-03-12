#include<iostream>
#include<exception>
#include"func.h"
#include<ctime>
#include<cstdlib>
#include<gurobi_c++.h>
#include<armadillo>
#include<iostream>

using namespace std;


NashGame* createCountry(
		const unsigned int n_followers,
		const vector<double> costs_quad,
		const vector<double> costs_lin,
		const vector<double> capacities,
		const double alpha, const double beta, // For the demand curve P = a-bQ
		const unsigned int LeadVars = 3 // One for tax and another for imposed cap and last for quantity imported
		)
{
	/// Check Error
	if(n_followers == 0) throw "Error in createCountry(). 0 Followers?";
	if (costs_lin.size()!=n_followers ||
			costs_quad.size() != n_followers ||
			capacities.size() != n_followers 
	   )
		throw "Error in createCountry(). Size Mismatch";
	if (alpha <= 0 || beta <=0 ) throw "Error in createCountry(). Invalid demand curve params";
	if (LeadVars < 3) throw "Error in createCountry(). At least 2 leader variables are there for cap and tax!";
	// Error checks over
	arma::sp_mat Q(1,1), C(1, LeadVars + n_followers - 1);
	/// Two constraints. One saying that you should be less than capacity
	// Another saying that you should be less than leader imposed cap!
	arma::sp_mat A(2, LeadVars + n_followers - 1), B(2, 1); 
	arma::vec c(1), b(2); 
	
	vector<QP_Param*> Players{};
	/// Create the QP_Param* for each follower
	for(unsigned int follower = 0; follower < n_followers; follower++)
	{
		c.fill(0); b.fill(0);
		A.zeros(); B.zeros(); C.zeros(); b.zeros(); Q.zeros(); c.zeros();
		QP_Param* Foll = new QP_Param();
		Q(0, 0) = costs_quad.at(follower) + 2*beta;
		c(0) = costs_lin.at(follower) - alpha;
		arma::mat Ctemp(1, LeadVars+n_followers-1); 
		Ctemp.fill(beta); Ctemp.tail_cols(2).fill(0); Ctemp.tail_cols(1) = 1;
		C = Ctemp;
		A(1, (n_followers-1)+2-1) = -1;
		B(0,0)=1; B(1,0) = 1;
		b(0) = capacities.at(follower);
		Foll->setMove(Q, C, A, B, c, b);
		Players.push_back(Foll);
	}
	arma::sp_mat MC(0, LeadVars+n_followers);
	arma::vec MCRHS(0, arma::fill::zeros);
	NashGame* N = new NashGame(Players, MC, MCRHS, LeadVars);
	return N;
}




int main()
{
	GRBEnv env = GRBEnv();
	GRBModel* model=nullptr;
	arma::sp_mat M;		 arma::vec q;		 perps Compl;

	NashGame *Country = createCountry(3, {3,2,1}, {1,2,3}, {1000, 1000, 1000}, 100, 5);
	bool Error{true};
	try{
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
		cout<<MyNashGame<<endl;
		cout<<"************************************"<<endl;
		MyNashGame.ConvexHull(&Aa, &b);
		cout<<"************************************"<<endl;
		// delete Aa; delete b;
	}
	catch(const char* e) { cout<<e<<endl; }
	catch(string e) { cout<<"String: "<<e<<endl; }
	catch(exception &e) { cout<<"Exception: "<<e.what()<<endl; }
	catch(GRBException &e) {cout<<"GRBException: "<<e.getErrorCode()<<": "<<e.getMessage()<<endl;}
	delete model;
	delete Country;
	b.save("b.txt", arma::arma_ascii);
	Aa.save("A.txt", arma::coord_ascii);
	cout<<Aa.n_rows<<", "<<Aa.n_cols<<endl;
	return 0;
}



/*
int BalasTest()
{
	vector<arma::sp_mat> Ai{};
	vector<arma::vec> bi{};


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
*/
