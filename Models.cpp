#include "func.h"
#include<vector>
#include<armadillo>
#include<iostream>
#include<gurobi_c++.h>

LCP* 
Models::createCountry(
		GRBEnv env,
		const unsigned int n_followers,
		const vector<double> costs_quad,
		const vector<double> costs_lin,
		const vector<double> capacities,
		const double alpha, const double beta, /// For the demand curve P = a-bQ
		const double import_limit, /// Negative number implies no limit
		const double export_limit,  /// Negative number implies no limit
		const double max_tax_perc,
		const unsigned int addnlLeadVars
		)
{
	const unsigned int LeadVars = 2 + 2*n_followers + addnlLeadVars;// two for quantity imported and exported, n for imposed cap and last n for tax
	/// @throw const char* - 0 followers
	if(n_followers == 0) throw "Error in createCountry(). 0 Followers?";
	/// @throw const char* - Not equal sizes of vectors
	if (costs_lin.size()!=n_followers ||
			costs_quad.size() != n_followers ||
			capacities.size() != n_followers 
	   )
		throw "Error in createCountry(). Size Mismatch";
	/// @throw const char* - Invalid alpha or beta value
	if (alpha <= 0 || beta <=0 ) throw "Error in createCountry(). Invalid demand curve params";
	// Error checks over
	arma::sp_mat Q(1,1), C(1, LeadVars + n_followers - 1);
	// Two constraints. One saying that you should be less than capacity
	// Another saying that you should be less than leader imposed cap!
	arma::sp_mat A(2, LeadVars + n_followers - 1), B(2, 1); 
	arma::vec c(1), b(2); 
	
	// Leader Constraints
	short int import_lim_cons{0}, export_lim_cons{0};
	if(import_limit < alpha && import_limit >= 0) import_lim_cons=1;
	if(export_limit >=0 ) export_lim_cons=1;

	arma::sp_mat LeadCons(import_lim_cons+export_lim_cons+n_followers, LeadVars+n_followers); arma::vec LeadRHS(import_lim_cons+export_lim_cons+n_followers, arma::fill::zeros);
	LeadCons.zeros();

	vector<QP_Param*> Players{};
	// Create the QP_Param* for each follower
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

		// Constraints of Tax limits!
		LeadCons(follower, n_followers+2+n_followers + follower) = 1;
		LeadRHS(follower) = max_tax_perc;
	}

	// Import limit - In more precise terms, everything that comes in minus everything that goes out should satisfy this limit
	if(import_lim_cons)
	{
		LeadCons(n_followers, n_followers) = 1;
		LeadCons(n_followers, n_followers+1) = -1;
		LeadRHS(n_followers) = import_limit;
	}	
	// Export limit - In more precise terms, everything that goes out minus everything that comes in should satisfy this limit
	if(export_lim_cons)
	{
		LeadCons(n_followers+import_lim_cons, n_followers+1) = 1;
		LeadCons(n_followers+import_lim_cons, n_followers) = -1;
		LeadRHS(n_followers) = export_limit;
	}


	arma::sp_mat MC(0, LeadVars+n_followers);
	arma::vec MCRHS(0, arma::fill::zeros);

	NashGame* N = new NashGame(Players, MC, MCRHS, LeadVars, LeadCons, LeadRHS);
	arma::sp_mat LeadConsReWrit = N->RewriteLeadCons();
	arma::sp_mat M; arma::vec q; perps Compl;
	N->FormulateLCP(M, q, Compl);
	LCP* country = new LCP(&env, M, q, Compl, LeadConsReWrit, LeadRHS);
	country->print();
	cout<<M.n_rows<<", "<<M.n_cols<<endl<<q.n_rows<<endl<<LeadConsReWrit.n_rows<<", "<<LeadConsReWrit.n_cols<<endl;
	delete N;
	return country;
}




