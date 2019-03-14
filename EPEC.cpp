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
		const unsigned int addnlLeadVars = 0,
		const double max_tax_perc = 0.30,
		const double import_limit = 10000,
		const double export_limit = 10000
		)
{
	/// Check Error
	const unsigned int LeadVars = 2 + 2*n_followers + addnlLeadVars;// two for quantity imported and exported, n for imposed cap and last n for tax
	if(n_followers == 0) throw "Error in createCountry(). 0 Followers?";
	if (costs_lin.size()!=n_followers ||
			costs_quad.size() != n_followers ||
			capacities.size() != n_followers 
	   )
		throw "Error in createCountry(). Size Mismatch";
	if (alpha <= 0 || beta <=0 ) throw "Error in createCountry(). Invalid demand curve params";
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

		arma::mat Ctemp(1, LeadVars+n_followers-1, arma::fill::zeros); 
		Ctemp.cols(0, n_followers-1).fill(beta); // First n-1 entries and 1 more entry is Beta
		Ctemp(0, n_followers) = -beta; // For q_exp

		Ctemp(0, (n_followers-1)+2+n_followers+follower  ) = 1; // q_{-i}, then import, export, then tilde q_i, then i-th tax

		C = Ctemp;
		A(1, (n_followers-1)+2 + follower) = -1;
		B(0,0)=1; B(1,0) = 1;
		b(0) = capacities.at(follower);
		Foll->setMove(Q, C, A, B, c, b);
		Players.push_back(Foll);
	}
	arma::sp_mat MC(0, LeadVars+n_followers);
	arma::vec MCRHS(0, arma::fill::zeros);
	//
	short int import_lim_cons{0}, export_lim_cons{0};
	if(import_limit < alpha) import_lim_cons=1;
	if(export_limit < alpha) export_lim_cons=1;

	arma::sp_mat LeadCons(2+n_followers, LeadVars+n_followers); arma::vec LeadRHS;

	NashGame* N = new NashGame(Players, MC, MCRHS, LeadVars, LeadCons, LeadRHS);
	return N;
}




int main()
{
	GRBEnv env = GRBEnv();
	GRBModel* model=nullptr;
	arma::sp_mat M;		 arma::vec q;		 perps Compl;

	NashGame *Country = createCountry(3, {3,2,1}, {1,2,3}, {1000, 1000, 1000}, 100, 5);
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



