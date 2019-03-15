#include "func.h"
#include<vector>
#include<armadillo>
#include<iostream>

NashGame* Models::createCountry(
		Models::LeadAllPar Params,
		const unsigned int addnlLeadVars
		)
{
	// Check Error
	const unsigned int LeadVars = 2 + 2*Params.n_followers + addnlLeadVars;// two for quantity imported and exported, n for imposed cap and last n for tax
	if(Params.n_followers == 0) throw "Error in createCountry(). 0 Followers?";
	if (Params.FollowerParam.costs_lin.size()!=Params.n_followers ||
			Params.FollowerParam.costs_quad.size() != Params.n_followers ||
			Params.FollowerParam.capacities.size() != Params.n_followers 
	   )
		throw "Error in createCountry(). Size Mismatch";
	if (Params.DemandParam.alpha <= 0 || Params.DemandParam.beta <=0 ) throw "Error in createCountry(). Invalid demand curve params";
	// Error checks over
	arma::sp_mat Q(1,1), C(1, LeadVars + Params.n_followers - 1);
	// Two constraints. One saying that you should be less than capacity
	// Another saying that you should be less than leader imposed cap!
	arma::sp_mat A(2, LeadVars + Params.n_followers - 1), B(2, 1); 
	arma::vec c(1), b(2); 
	
	// Leader Constraints
	short int import_lim_cons{0}, export_lim_cons{0};
	if(Params.LeaderParam.import_limit < Params.DemandParam.alpha && Params.LeaderParam.import_limit >= 0) import_lim_cons=1;
	if(Params.LeaderParam.export_limit >=0 ) export_lim_cons=1;

	arma::sp_mat LeadCons(import_lim_cons+export_lim_cons+Params.n_followers, LeadVars+Params.n_followers); arma::vec LeadRHS(import_lim_cons+export_lim_cons+Params.n_followers, arma::fill::zeros);
	LeadCons.zeros();

	vector<QP_Param*> Players{};
	// Create the QP_Param* for each follower
	for(unsigned int follower = 0; follower < Params.n_followers; follower++)
	{
		c.fill(0); b.fill(0);
		A.zeros(); B.zeros(); C.zeros(); b.zeros(); Q.zeros(); c.zeros();
		QP_Param* Foll = new QP_Param();
		Q(0, 0) = Params.FollowerParam.costs_quad.at(follower) + 2*Params.DemandParam.beta;
		c(0) = Params.FollowerParam.costs_lin.at(follower) - Params.DemandParam.alpha;

		arma::mat Ctemp(1, LeadVars+Params.n_followers-1, arma::fill::zeros); 
		Ctemp.cols(0, Params.n_followers-1).fill(Params.DemandParam.beta); // First n-1 entries and 1 more entry is Beta
		Ctemp(0, Params.n_followers) = -Params.DemandParam.beta; // For q_exp

		Ctemp(0, (Params.n_followers-1)+2+Params.n_followers+follower  ) = 1; // q_{-i}, then import, export, then tilde q_i, then i-th tax

		C = Ctemp;
		A(1, (Params.n_followers-1)+2 + follower) = -1;
		B(0,0)=1; B(1,0) = 1;
		b(0) = Params.FollowerParam.capacities.at(follower);
		Foll->setMove(Q, C, A, B, c, b);
		Players.push_back(Foll);

		// Constraints of Tax limits!
		LeadCons(follower, Params.n_followers+2+Params.n_followers + follower) = 1;
		LeadRHS(follower) = Params.LeaderParam.max_tax_perc;
	}

	// Import limit - In more precise terms, everything that comes in minus everything that goes out should satisfy this limit
	if(import_lim_cons)
	{
		LeadCons(Params.n_followers, Params.n_followers) = 1;
		LeadCons(Params.n_followers, Params.n_followers+1) = -1;
		LeadRHS(Params.n_followers) = Params.LeaderParam.import_limit;
	}	
	// Export limit - In more precise terms, everything that goes out minus everything that comes in should satisfy this limit
	if(export_lim_cons)
	{
		LeadCons(Params.n_followers+import_lim_cons, Params.n_followers+1) = 1;
		LeadCons(Params.n_followers+import_lim_cons, Params.n_followers) = -1;
		LeadRHS(Params.n_followers) = Params.LeaderParam.export_limit;
	}


	arma::sp_mat MC(0, LeadVars+Params.n_followers);
	arma::vec MCRHS(0, arma::fill::zeros);

	NashGame* N = new NashGame(Players, MC, MCRHS, LeadVars, LeadCons, LeadRHS);
	// NashGame* N = new NashGame(Players, MC, MCRHS, LeadVars); 
	return N;
}


ostream& Models::operator<<(ostream& ost, const Models::FollPar P)
{
	ost<<"Follower Parameters: "<<endl;
	ost<<"********************"<<endl;
	ost<<"Linear Costs: \t\t\t\t";
	for(auto a:P.costs_lin) ost<<a<<"\t";
	ost<<endl<<"Quadratic costs: \t\t\t";
	for(auto a:P.costs_quad) ost<<a<<"\t";
	ost<<endl<<"Production capacities: \t\t\t";
	for(auto a:P.capacities) ost<<(a<0?string("Inf"):to_string(a))<<"\t";
	ost<<endl;
	return ost;
}

ostream& Models::operator<<(ostream& ost, const Models::DemPar P)
{
	ost<<"Demand Parameters: "<<endl;
	ost<<"******************"<<endl;
	ost<<"Price\t\t =\t\t "<<P.alpha<<"\t\t-\t\t"<<P.beta<<"\tx\t Quantity"<<endl;
	return ost;
}
ostream& Models::operator<<(ostream& ost, const Models::LeadPar P)
{
	ost<<"Leader Parameters: "<<endl;
	ost<<"******************"<<endl;
	ost<<"Export Limit: \t\t\t"<<(P.export_limit<0?string("Inf"):to_string(P.export_limit));
	ost<<endl;
	ost<<"Import Limit: \t\t\t"<<(P.import_limit<0?string("Inf"):to_string(P.import_limit));
	ost<<endl;
	ost<<"Maximum tax percentage: \t"<<P.max_tax_perc;
	ost<<endl;
	return ost;
}


ostream& Models::operator<<(ostream& ost, const Models::LeadAllPar P)
{
	ost<<"\n\n";
	ost<<"***************************"<<"\n "<<"\n";
	ost<<"Leader Complete Description"<<"\n "<<"\n";
	ost<<"***************************"<<"\n "<<"\n";
	ost<<"Number of followers: \t\t\t"<<P.n_followers<<"\n "<<"\n";
	ost<<P.LeaderParam<<P.FollowerParam<<P.DemandParam<<"\n";
	ost<<"***************************"<<"\n"<<"\n";
	return ost;
}

	
