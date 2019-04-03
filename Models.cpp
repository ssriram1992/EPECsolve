#include "func.h"
#include<map>
#include<memory>
#include<vector>
#include<armadillo>
#include<iostream>
#include<gurobi_c++.h>

ostream& 
Models::operator<<(ostream& ost, const Models::FollPar P)
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

ostream& 
Models::operator<<(ostream& ost, const Models::DemPar P)
{
	ost<<"Demand Parameters: "<<endl;
	ost<<"******************"<<endl;
	ost<<"Price\t\t =\t\t "<<P.alpha<<"\t-\t"<<P.beta<<"  x   Quantity"<<endl;
	return ost;
}

ostream& 
Models::operator<<(ostream& ost, const Models::LeadPar P)
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

ostream& 
Models::operator<<(ostream& ost, const Models::LeadAllPar P)
{
	ost<<"\n\n";
	ost<<"***************************"<<"\n";
	ost<<"Leader Complete Description"<<"\n";
	ost<<"***************************"<<"\n"<<"\n";
	ost<<"Number of followers: \t\t\t"<<P.n_followers<<"\n "<<"\n";
	ost<<endl<<P.LeaderParam<<endl<<P.FollowerParam<<endl<<P.DemandParam<<"\n";
	ost<<"***************************"<<"\n"<<"\n";
	return ost;
}


ostream& 
Models::operator<<(ostream& ost, const Models::LeaderVars l)
{
	switch(l)
	{
		case Models::LeaderVars::FollowerStart:
			ost<<"Models::LeaderVars::FollowerStart";
			break;
		case Models::LeaderVars::NetImport:
			ost<<"Models::LeaderVars::NetImport";
			break;
		case Models::LeaderVars::NetExport:
			ost<<"Models::LeaderVars::NetExport";
			break;
		case Models::LeaderVars::CountryImport:
			ost<<"Models::LeaderVars::CountryImport";
			break;
		case Models::LeaderVars::Tax:
			ost<<"Models::LeaderVars::Tax";
			break;
		case Models::LeaderVars::Caps:
			ost<<"Models::LeaderVars::Caps";
			break;
		case Models::LeaderVars::AddnVar:
			ost<<"Models::LeaderVars::AddnVar";
			break;
		case Models::LeaderVars::ConvHullDummy:
			ost<<"Models::LeaderVars::ConvHullDummy";
			break;
		case Models::LeaderVars::End:
			ost<<"Models::LeaderVars::End";
			break;
		default:
			cerr<<"Incorrect argument to ostream& operator<<(ostream& ost, const LeaderVars l)";
	};
	return ost;
}



bool 
Models::EPEC::ParamValid(const LeadAllPar& Params ///< Object whose validity is to be tested
		) const
/**
 * @brief Checks the Validity of Models::LeadAllPar object
 * @details Checks the following:
 * 	-	Size of FollowerParam.costs_lin, FollowerParam.costs_quad, FollowerParam.capacities, FollowerParam.emission_costs are all equal to @p Params.n_followers
 * 	-	@p DemandParam.alpha and @p DemandParam.beta are greater than zero
 * 	-	@p name is not empty
 * 	-	@p name does not match with the name of any other existing countries in the EPEC object.
 */
{
	if(Params.n_followers == 0) throw "Error in EPEC::ParamValid(). 0 Followers?";
	if (Params.FollowerParam.costs_lin.size()			!= Params.n_followers ||
			Params.FollowerParam.costs_quad.size() 		!= Params.n_followers ||
			Params.FollowerParam.capacities.size() 		!= Params.n_followers ||
			Params.FollowerParam.emission_costs.size()	!= Params.n_followers
	   )
		throw "Error in EPEC::ParamValid(). Size Mismatch";
	if (Params.DemandParam.alpha <= 0 || Params.DemandParam.beta <=0 ) throw "Error in EPEC::ParamValid(). Invalid demand curve params";
	// Country should have a name!
	if(Params.name=="") 
		throw "Error in EPEC::ParamValid(). Country name empty";
	// Country should have a unique name
	for(const auto &p:this->AllLeadPars)
		if(Params.name.compare(p.name) == 0) // i.e., if the strings are same
			throw "Error in EPEC::ParamValid(). Country name repetition";
	return true;
}

void 
Models::EPEC::make_LL_QP(const LeadAllPar& Params, 	///< The Parameters object
		const unsigned int follower, 				///< Which follower's QP has to be made?
		Game::QP_Param* Foll, 						///< Non-owning pointer to the Follower QP_Param object
		const Models::LeadLocs& Loc					///< LeadLocs object for accessing different leader locations.
		) const noexcept
/**
 * @brief Makes Lower Level Quadratic Programs
 * @details Sets the constraints and objective for the lower level problem (i.e., the follower)
 */
{
		const unsigned int LeadVars = Loc.at(Models::LeaderVars::End) - Params.n_followers;
		arma::sp_mat Q(1,1), C(1, LeadVars + Params.n_followers - 1);
		// Two constraints. One saying that you should be less than capacity
		// Another saying that you should be less than leader imposed cap!
		arma::sp_mat A(2, Loc.at(Models::LeaderVars::End) - 1), B(2, 1); 
		arma::vec c(1), b(2); 
		c.fill(0); b.fill(0);
		A.zeros(); B.zeros(); C.zeros(); b.zeros(); Q.zeros(); c.zeros();
		// Objective
		Q(0, 0) = Params.FollowerParam.costs_quad.at(follower) + 2*Params.DemandParam.beta;
		c(0) = Params.FollowerParam.costs_lin.at(follower) - Params.DemandParam.alpha;

		arma::mat Ctemp(1, Loc.at(Models::LeaderVars::End)-1, arma::fill::zeros); 
		Ctemp.cols(0, Params.n_followers-1).fill(Params.DemandParam.beta); // First n-1 entries and 1 more entry is Beta
		Ctemp(0, Params.n_followers) = -Params.DemandParam.beta; // For q_exp

		Ctemp(0, (Params.n_followers-1)+2+Params.n_followers+follower  ) = 1; // q_{-i}, then import, export, then tilde q_i, then i-th tax

		C = Ctemp;
		A(1, (Params.n_followers-1)+2 + follower) = -1;
		B(0,0)=1; B(1,0) = 1;
		b(0) = Params.FollowerParam.capacities.at(follower); 

		Foll->set(std::move(Q), std::move(C), std::move(A), std::move(B), std::move(c), std::move(b));
}

void 
Models::EPEC::make_LL_LeadCons(arma::sp_mat &LeadCons, arma::vec &LeadRHS,
			/// All country specific parameters
			const LeadAllPar& Params,
			/// Location of variables
			const Models::LeadLocs& Loc,
			/// Does a constraint on import limit exist or no limit?
			const unsigned int import_lim_cons,
			/// Does a constraint on export limit exist or no limit?
			const unsigned int export_lim_cons,
			/// Does a constraint on price limit exist or no limit?
			const unsigned int price_lim_cons
			) const noexcept
/**
 * Makes the leader level constraints for a country.
 * The constraints added are as follows:
 * @f{eqnarray}{
 *	q^{import} - q^{export} &\leq& \bar{q^{import}}\\
 *	q^{export} - q^{import} &\leq& \bar{q^{export}}\\
 *	\alpha - \beta\left(q^{import} - q^{export} + \sum_i q_i \right) &\leq& \bar{\pi}\\
 *	q^{export} &\leq& \sum_i q_i +q^{import} 
 * @f}
 * Here @f$\bar{q^{import}}@f$ and @f$\bar{q^{export}}@f$ denote the net import limit and export limit respectively. @f$\bar\pi@f$ is the maximum local price that the government desires to have.
 *
 * The first two constraints above limit net imports and exports respectively. The third constraint limits local price. These constraints are added only if the RHS parameters are given as non-negative value. A default value of -1 to any of these parameters (given in Models::LeadAllPar @p Params object) ensures that these constraints are not added. The last constraint is <i>always</i> added. It ensures that the country does not export more than what it has produced + imported!
 */
{
	for(unsigned int follower = 0; follower < Params.n_followers; follower++)
	{
		// Constraints of Tax limits!
		LeadCons(follower, Loc.at(Models::LeaderVars::Tax)+follower) = 1;
		LeadRHS(follower) = Params.LeaderParam.max_tax_perc;
	}
	// Export - import <= Local Production
	for (unsigned int i=0;i<Params.n_followers;i++) LeadCons.at(Params.n_followers, i) = -1;
	LeadCons.at(Params.n_followers, Loc.at(Models::LeaderVars::NetExport)) = 1;
	LeadCons.at(Params.n_followers, Loc.at(Models::LeaderVars::NetImport)) = -1;
	// Import limit - In more precise terms, everything that comes in minus everything that goes out should satisfy this limit
	if(import_lim_cons)
	{
		LeadCons(Params.n_followers+1, Loc.at(Models::LeaderVars::NetImport)) = 1;
		LeadCons(Params.n_followers+1, Loc.at(Models::LeaderVars::NetExport)) = -1;
		LeadRHS(Params.n_followers+1) = Params.LeaderParam.import_limit;
	}	
	// Export limit - In more precise terms, everything that goes out minus everything that comes in should satisfy this limit
	if(export_lim_cons)
	{
		LeadCons(Params.n_followers+1+import_lim_cons, Loc.at(Models::LeaderVars::NetExport)) = 1;
		LeadCons(Params.n_followers+1+import_lim_cons, Loc.at(Models::LeaderVars::NetImport)) = -1;
		LeadRHS(Params.n_followers+1) = Params.LeaderParam.export_limit;
	}
	if(price_lim_cons)
	{
		for (unsigned int i=0;i<Params.n_followers;i++)
			LeadCons.at(Params.n_followers+1+import_lim_cons+export_lim_cons, i) = -Params.DemandParam.beta;
		LeadCons.at(Params.n_followers+1+import_lim_cons+export_lim_cons, Loc.at(Models::LeaderVars::NetImport)) = -Params.DemandParam.beta;
		LeadCons.at(Params.n_followers+1+import_lim_cons+export_lim_cons, Loc.at(Models::LeaderVars::NetExport)) = Params.DemandParam.beta;
		LeadRHS.at(Params.n_followers+1+import_lim_cons+export_lim_cons) = Params.LeaderParam.price_limit - Params.DemandParam.alpha;
	}
}

Models::EPEC& 
Models::EPEC::addCountry(
		Models::LeadAllPar Params,
		const unsigned int addnlLeadVars
		)
	/**
	 *  A Nash cournot game is played among the followers, for the leader-decided values of import export, caps and taxations on all players. The total quantity used in the demand equation is the sum of quantity produced by all followers + any import - any export.  
	 */
	/**
	 * @details Use \f$l_i\f$ to denote the \f$i\f$-th element in `costs_lin` and \f$q_i\f$ for the \f$i\f$-th element in `costs_quad`. Then to produce quantity \f$x_i\f$, the \f$i\f$-th producer's cost will be 
	 * \f[ l_ix_i + \frac{1}{2}q_ix_i^2 \f]
	 * In addition to this, the leader may impose "tax", which could increase \f$l_i\f$ for each player.
	 *
	 * Total quantity in the market is given by sum of quantities produced by all producers adjusted by imports and exports
	 * \f[{Total\quad  Quantity} = \sum_i x_i + x_{imp} - x_{exp} \f]
	 * The demand curve in the market is given by
	 * \f[{Price} = a-b({Total\quad  Quantity})\f]
	 *
	 * Each follower is also constrained by a maximum production capacity her infrastructure allows. And each follower is constrained by a cap on their production, that is imposed by the leader.
	 *
	 * Each follower decides \f$x_i\f$ noncooperatively maximizing profits.
	 *
	 * The leader decides quantity imported \f$q_{imp}\f$, quantity exported \f$q_{exp}\f$, cap on each player, \f$\tilde{x_i}\f$, and the tax for each player \f$t_i\f$.
	 *
	 * The leader is also constrained to not export or import anything more than the limits set by `export_limit` and `import_limit`. A negative value to these input variables imply that there is no such limit.
	 * 
	 * Similarly the leader cannot also impose tax on any player greater than what is dictated by the input variable `max_tax_perc`.
	 *
	 * @return Pointer to LCP object dynamically created using `new`. 
	 */
{
	bool noError=false;
	try { noError = this->ParamValid(Params); }
	catch(const char* e) { cerr<<"Error in Models::EPEC::addCountry: "<<e<<endl; }
	catch(string e) { cerr<<"String: Error in Models::EPEC::addCountry: "<<e<<endl; }
	catch(exception &e) { cerr<<"Exception: Error in Models::EPEC::addCountry: "<<e.what()<<endl; }
	if(!noError) return *this;

	const unsigned int LeadVars = 2 + 2*Params.n_followers + addnlLeadVars;// two for quantity imported and exported, n for imposed cap and last n for tax

	LeadLocs Loc;
	Loc[Models::LeaderVars::FollowerStart] = 0;
	Loc[Models::LeaderVars::NetImport] = Loc[Models::LeaderVars::FollowerStart] + Params.n_followers;
	Loc[Models::LeaderVars::NetExport] = Loc[Models::LeaderVars::NetImport] + 1;
	Loc[Models::LeaderVars::Caps] = Loc[Models::LeaderVars::NetExport] + 1;
	Loc[Models::LeaderVars::Tax] = Loc[Models::LeaderVars::Caps] + Params.n_followers;
	Loc[Models::LeaderVars::End] = Loc[Models::LeaderVars::Tax] + Params.n_followers;

	Locations.push_back(Loc);
	
	// Loc[Models::LeaderVars::AddnVar] = 1;
	
	// Leader Constraints
	short int import_lim_cons{0}, export_lim_cons{0}, price_lim_cons{0};
	if(Params.LeaderParam.import_limit >= 0) import_lim_cons = 1;
	if(Params.LeaderParam.export_limit >= 0) export_lim_cons = 1;
	if(Params.LeaderParam.price_limit  >  0) price_lim_cons  = 1; 

	arma::sp_mat LeadCons(import_lim_cons+	// Import limit constraint
			export_lim_cons+				// Export limit constraint
			price_lim_cons+					// Price limit constraint
			Params.n_followers+				// Tax limit constraint
			1, 								// Export - import <= Domestic production
			Loc[Models::LeaderVars::End]
			);
	arma::vec LeadRHS(import_lim_cons+
			export_lim_cons+
			price_lim_cons+
			Params.n_followers+
			1, arma::fill::zeros);

	vector<shared_ptr<Game::QP_Param>> Players{};
	// Create the QP_Param* for each follower
	try
	{
		for(unsigned int follower = 0; follower < Params.n_followers; follower++)
		{
			shared_ptr<Game::QP_Param> Foll(new Game::QP_Param(this->env));
			this->make_LL_QP(Params, follower, Foll.get(), Loc);
			Players.push_back(Foll); 
		}
	}
	catch(const char* e) { cerr<<e<<endl;throw; }
	catch(string e) { cerr<<"String in Models::EPEC::addCountry : "<<e<<endl;throw; }
	catch(GRBException &e) {cerr<<"GRBException in Models::EPEC::addCountry : "<<e.getErrorCode()<<": "<<e.getMessage()<<endl;throw;}
	catch(exception &e) { cerr<<"Exception in Models::EPEC::addCountry : "<<e.what()<<endl;throw; }

	// Make Leader Constraints
	try
	{
		this->make_LL_LeadCons(LeadCons, LeadRHS, Params, Loc, import_lim_cons, export_lim_cons);
	}
	catch(const char* e) { cerr<<e<<endl;throw; }
	catch(string e) { cerr<<"String: "<<e<<endl;throw; }
	catch(GRBException &e) {cerr<<"GRBException: "<<e.getErrorCode()<<": "<<e.getMessage()<<endl;throw;}
	catch(exception &e) { cerr<<"Exception in Models::EPEC::addCountry : "<<e.what()<<endl;throw; }

	// Lower level Market clearing constraints - empty
	arma::sp_mat MC(0, LeadVars+Params.n_followers);
	arma::vec MCRHS(0, arma::fill::zeros);

	auto N = make_shared<Game::NashGame>(Players, MC, MCRHS, LeadVars, LeadCons, LeadRHS);
	this->name2nos[Params.name] = this->countriesLL.size();
	this->countriesLL.push_back(N);
	this->LeadConses.push_back(N->RewriteLeadCons());
	this->AllLeadPars.push_back(Params);
	nCountr++;


	return *this;
}

Models::EPEC& 
Models::EPEC::addTranspCosts(const arma::sp_mat& costs)
{
	try
	{
		if(this->nCountries!=costs.n_rows || this->nCountries!=costs.n_cols) throw "Error in EPEC::addTranspCosts. Invalid size of Q";
		else this->TranspCosts = arma::sp_mat(costs);
		this->TranspCosts.diag().zeros(); 		// Doesn't make sense for it to have a nonzero diagonal!
	}
	catch(const char* e) { cerr<<e<<endl;throw; }
	catch(string e) { cerr<<"String in Models::EPEC::addTranspCosts : "<<e<<endl;throw; }
	catch(GRBException &e) {cerr<<"GRBException in Models::EPEC::addTranspCosts : "<<e.getErrorCode()<<": "<<e.getMessage()<<endl;throw;}
	catch(exception &e) { cerr<<"Exception in Models::EPEC::addTranspCosts : "<<e.what()<<endl;throw; }

	return *this;
}

const 
Models::EPEC& 
Models::EPEC::
finalize() // Incomplete
{
	try
	{
		/* 
		 * Below for loop adds space for each country's quantity imported from variable
		 */
		this->nImportMarkets = vector<unsigned int> (this->nCountries);
		for(unsigned int i=0; i<this->nCountries; i++)
			this->add_Leaders_tradebalance_constraints(i);

		/*
		 * Now we keep track of where each country's variables start
		 */
		this->computeLeaderLocations(true);
		
		this->MC_QP = vector<shared_ptr<Game::QP_Param>>(nCountr);
		this->add_Dummy_All_Lead();
		for(unsigned int i=0; i<this->nCountries; i++) // To add the corresponding Market Clearing constraint
			this->make_MC_leader(i); // This function is bogus and incomplete now.
	}
	catch(const char* e) { cerr<<e<<endl;throw; }
	catch(string e) { cerr<<"String in Models::EPEC::finalize : "<<e<<endl;throw; }
	catch(GRBException &e) {cerr<<"GRBException in Models::EPEC::finalize : "<<e.getErrorCode()<<": "<<e.getMessage()<<endl;throw;}
	catch(exception &e) { cerr<<"Exception in Models::EPEC::finalize : "<<e.what()<<endl;throw; }
	return *this;
}

void 
Models::EPEC::add_Leaders_tradebalance_constraints(const unsigned int i)
{ 
	if (i>=this->nCountries) throw string("Error in Models::EPEC::add_Leaders_tradebalance_constraints. Bad argument");
	int nImp = 0;
	// Counts the number of countries from which the current country imports
	for(auto val=TranspCosts.begin_col(i); val!=TranspCosts.end_col(i); ++val) nImp++;
	// substitutes that answer to nImportMarkets at the current position
	this->nImportMarkets.at(i) = (nImp);
	// Adding the constraint that the sum of imports from all countries equals total imports
	arma::vec a(nImp + this->countriesLL.at(i)->getNprimals(), arma::fill::zeros);
	const auto n_followers = this->AllLeadPars.at(i).n_followers;
	a.at(n_followers) = -1; 
	a.tail(nImp) = 1;
	this->countriesLL.at(i)->addDummy(nImp).addLeadCons(a, 0).addLeadCons(-a,0);
	Locations.at(i)[Models::LeaderVars::CountryImport] = Locations.at(i).at(Models::LeaderVars::End);
	Locations.at(i).at(Models::LeaderVars::End) += nImp;
}

void 
Models::EPEC::make_MC_leader(unsigned int i)
{
	if (i>=this->nCountries) throw string("Error in Models::EPEC::add_Leaders_tradebalance_constraints. Bad argument");
	try
	{
		arma::sp_mat Q(1,1);
		arma::sp_mat A, B, C; // To define well
		arma::vec c, b; 	// To define well
		this->MC_QP.at(i) = make_shared<Game::QP_Param>(this->env);
		// Note Q = {0}, the MC problem has no constraints. So A=B=b={}. 
		// this->MC_QP.at(i).get()->set({0},std::move(C), {}, {}, {0}, {});
	}
	catch(const char* e) { cerr<<e<<endl;throw; }
	catch(string e) { cerr<<"String in Models::EPEC::make_MC_leader : "<<e<<endl;throw; }
	catch(GRBException &e) {cerr<<"GRBException in Models::EPEC::make_MC_leader : "<<e.getErrorCode()<<": "<<e.getMessage()<<endl;throw;}
	catch(exception &e) { cerr<<"Exception in Models::EPEC::make_MC_leader : "<<e.what()<<endl;throw; }
}

bool 
Models::EPEC::dataCheck( 
			bool chkAllLeadPars,
			bool chkcountriesLL,
			bool chkMC_QP,
			bool chkLeadConses,
			bool chkLeadRHSes,
			bool chknImportMarkets,
			bool chkLocations,
			bool chkLeaderLocations
		) const
{
	if (!chkAllLeadPars && AllLeadPars.size() 			!= this->nCountries) return false;
	if (!chkcountriesLL && countriesLL.size() 			!= this->nCountries) return false;
	if (!chkMC_QP && MC_QP.size() 						!= this->nCountries) return false;
	if (!chkLeadConses && LeadConses.size() 			!= this->nCountries) return false;
	if (!chkLeadRHSes && LeadRHSes.size() 				!= this->nCountries) return false;
	if (!chknImportMarkets && nImportMarkets.size() 	!= this->nCountries) return false;
	if (!chkLocations && Locations.size() 				!= this->nCountries) return false;
	if (!chkLeaderLocations && LeaderLocations.size() 	!= this->nCountries) return false; 
	return true;
}


void 
Models::EPEC::add_Dummy_All_Lead()
{
	if(!this->dataCheck())
		throw string("Error in Models::EPEC::add_Dummy_All_Lead: dataCheck() failed!");
}

void
Models::EPEC::computeLeaderLocations(bool addSpaceForMC)
{
	this->LeaderLocations = vector<unsigned int> (this->nCountries);
	this->LeaderLocations.at(0) = 0;
	for(unsigned int i=1; i<this->nCountries; i++)
		this->LeaderLocations.at(i) = this->LeaderLocations.at(i-1) + this->Locations.at(i-1).at(Models::LeaderVars::End) 
			+ (addSpaceForMC?1:0);
}


unsigned int 
Models::EPEC::getPosition(unsigned int countryCount, Models::LeaderVars var) const
{
	if(countryCount > this->nCountries) throw string("Error in Models::EPEC::getPosition: Bad Country Count");
	return this->LeaderLocations.at(countryCount) + this->Locations.at(countryCount).at(var);
}


unsigned int 
Models::EPEC::getPosition(string countryName, Models::LeaderVars var) const
{ 
	return this->getPosition(name2nos.at(countryName), var);
}



/*
Game::LCP* 
Models::EPEC::playCountry(vector<Game::LCP*> countries) 
{
	auto Pi = &this->AllLeadPars;
	vector<unsigned int> LeadVars(this->nCountries, 0);
	for(unsigned int i=0;i<this->nCountries;i++)
		LeadVars.at(i) = 2 + 2*Pi->at(i).n_followers + Pi->at(i).n_followers;		// two for quantity imported and exported, n for imposed cap and last n for tax and finally n_follower number of follower variables
	return nullptr;
} 

*/
