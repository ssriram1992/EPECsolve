// #include "models.h"
#include"epecsolve.h"
#include<iomanip>
#include<map>
#include<memory>
#include<vector>
#include<armadillo>
#include<iostream>
#include<gurobi_c++.h>


ostream &
Models::operator<<(ostream &ost, const Models::prn l) {
    switch (l) {
        case Models::prn::label:
            ost << std::left << std::setw(50);
            break;
        case Models::prn::val:
            ost << std::right << std::setprecision(2) << std::setw(16) << std::fixed;
            break;
        default:
            break;
    }
    return ost;
}

ostream &
Models::operator<<(ostream &ost, const Models::FollPar P) {
    ost << "Follower Parameters: " << endl;
    ost << "********************" << endl;
    ost << Models::prn::label << "Linear Costs" << ":\t";
    for (auto a:P.costs_lin) ost << Models::prn::val << a;
    ost << endl << Models::prn::label << "Quadratic costs" << ":\t";
    for (auto a:P.costs_quad) ost << Models::prn::val << a;
    ost << endl << Models::prn::label << "Production capacities" << ":\t";
    for (auto a:P.capacities) ost << Models::prn::val << (a < 0 ? std::numeric_limits<double>::infinity() : a);
    ost << endl;
    return ost;
}

ostream &
Models::operator<<(ostream &ost, const Models::DemPar P) {
    ost << "Demand Parameters: " << endl;
    ost << "******************" << endl;
    ost << "Price\t\t =\t\t " << P.alpha << "\t-\t" << P.beta << "  x   Quantity" << endl;
    return ost;
}

ostream &
Models::operator<<(ostream &ost, const Models::LeadPar P) {
    ost << "Leader Parameters: " << endl;
    ost << "******************" << endl;
    ost << std::fixed;
    ost << Models::prn::label << "Export Limit" << ":" << Models::prn::val
        << (P.export_limit < 0 ? std::numeric_limits<double>::infinity() : P.export_limit);
    ost << endl;
    ost << Models::prn::label << "Import Limit" << ":" << Models::prn::val
        << (P.import_limit < 0 ? std::numeric_limits<double>::infinity() : P.import_limit);
    ost << endl;
    ost << Models::prn::label << "Maximum Tax" << ":" << Models::prn::val
        << (P.max_tax < 0 ? std::numeric_limits<double>::infinity() : P.max_tax);
    ost << endl;
    ost << Models::prn::label << "Price limit" << ":" << Models::prn::val
        << (P.price_limit < 0 ? std::numeric_limits<double>::infinity() : P.price_limit);
    ost << endl;
    return ost;
}

ostream &
Models::operator<<(ostream &ost, const Models::LeadAllPar P) {
    ost << "\n\n";
    ost << "***************************" << "\n";
    ost << "Leader Complete Description" << "\n";
    ost << "***************************" << "\n" << "\n";
    ost << Models::prn::label << "Number of followers" << ":" << Models::prn::val << P.n_followers << "\n " << "\n";
    ost << endl << P.LeaderParam << endl << P.FollowerParam << endl << P.DemandParam << "\n";
    ost << "***************************" << "\n" << "\n";
    return ost;
}


ostream &
Models::operator<<(ostream &ost, const Models::LeaderVars l) {
    switch (l) {
        case Models::LeaderVars::FollowerStart:
            ost << "Models::LeaderVars::FollowerStart";
            break;
        case Models::LeaderVars::NetImport:
            ost << "Models::LeaderVars::NetImport";
            break;
        case Models::LeaderVars::NetExport:
            ost << "Models::LeaderVars::NetExport";
            break;
        case Models::LeaderVars::CountryImport:
            ost << "Models::LeaderVars::CountryImport";
            break;
        case Models::LeaderVars::Caps:
            ost << "Models::LeaderVars::Caps";
            break;
        case Models::LeaderVars::Tax:
            ost << "Models::LeaderVars::Tax";
            break;
        case Models::LeaderVars::DualVar:
            ost << "Models::LeaderVars::DualVar";
            break;
        case Models::LeaderVars::ConvHullDummy:
            ost << "Models::LeaderVars::ConvHullDummy";
            break;
        case Models::LeaderVars::End:
            ost << "Models::LeaderVars::End";
            break;
        default:
            cerr << "Incorrect argument to ostream& operator<<(ostream& ost, const LeaderVars l)";
    };
    return ost;
}


bool
Models::EPEC::ParamValid(const LeadAllPar &Params ///< Object whose validity is to be tested
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
    if (Params.n_followers == 0) throw "Error in EPEC::ParamValid(). 0 Followers?";
    if (Params.FollowerParam.costs_lin.size() != Params.n_followers ||
        Params.FollowerParam.costs_quad.size() != Params.n_followers ||
        Params.FollowerParam.capacities.size() != Params.n_followers ||
        Params.FollowerParam.emission_costs.size() != Params.n_followers
            )
        throw "Error in EPEC::ParamValid(). Size Mismatch";
    if (Params.DemandParam.alpha <= 0 || Params.DemandParam.beta <= 0)
        throw "Error in EPEC::ParamValid(). Invalid demand curve params";
    // Country should have a name!
    if (Params.name == "")
        throw "Error in EPEC::ParamValid(). Country name empty";
    // Country should have a unique name
    for (const auto &p:this->AllLeadPars)
        if (Params.name.compare(p.name) == 0) // i.e., if the strings are same
            throw "Error in EPEC::ParamValid(). Country name repetition";
    return true;
}

void
Models::EPEC::make_LL_QP(const LeadAllPar &Params,    ///< The Parameters object
                         const unsigned int follower,                ///< Which follower's QP has to be made?
                         Game::QP_Param *Foll,                        ///< Non-owning pointer to the Follower QP_Param object
                         const Models::LeadLocs &Loc                    ///< LeadLocs object for accessing different leader locations.
) const noexcept
/**
 * @brief Makes Lower Level Quadratic Programs
 * @details Sets the constraints and objective for the lower level problem (i.e., the follower)
 */
{
    const unsigned int LeadVars = Loc.at(Models::LeaderVars::End) - Params.n_followers;
    arma::sp_mat Q(1, 1), C(1, LeadVars + Params.n_followers - 1);
    // Two constraints. One saying that you should be less than capacity
    // Another saying that you should be less than leader imposed cap!
    arma::sp_mat A(2, Loc.at(Models::LeaderVars::End) - 1), B(2, 1);
    arma::vec c(1), b(2);
    c.fill(0);
    b.fill(0);
    A.zeros();
    B.zeros();
    C.zeros();
    b.zeros();
    Q.zeros();
    c.zeros();
    // Objective
    Q(0, 0) = Params.FollowerParam.costs_quad.at(follower) + 2 * Params.DemandParam.beta;
    c(0) = Params.FollowerParam.costs_lin.at(follower) - Params.DemandParam.alpha;

    arma::mat Ctemp(1, Loc.at(Models::LeaderVars::End) - 1, arma::fill::zeros);
    Ctemp.cols(0, Params.n_followers - 1).fill(Params.DemandParam.beta); // First n-1 entries and 1 more entry is Beta
    Ctemp(0, Params.n_followers) = -Params.DemandParam.beta; // For q_exp

    Ctemp(0, (Params.n_followers - 1) + 2 + Params.n_followers +
             follower) = 1; // q_{-i}, then import, export, then tilde q_i, then i-th tax

    C = Ctemp;
    //A(1, (Params.n_followers - 1) + 2 + follower) = 0;
    //Produce positive (zero) quantities and less than the cap
    B(0, 0) = 1;
    B(1, 0) = -1;
    b(0) = Params.FollowerParam.capacities.at(follower);
    b(1) = 0; // - Params.FollowerParam.capacities.at(follower)*0.05;

    Foll->set(std::move(Q), std::move(C), std::move(A), std::move(B), std::move(c), std::move(b));
}

void Models::EPEC::make_LL_LeadCons(
        arma::sp_mat &LeadCons,                ///< The LHS matrix of leader constraints (for output)
        arma::vec &LeadRHS,                    ///< RHS vector for leader constraints (for output)
        const LeadAllPar &Params,            ///< All country specific parameters
        const Models::LeadLocs &Loc,        ///< Location of variables
        const unsigned int import_lim_cons, ///< Does a constraint on import limit exist or no limit?
        const unsigned int export_lim_cons, ///< Does a constraint on export limit exist or no limit?
        const unsigned int price_lim_cons,   ///< Does a constraint on price limit exist or no limit?
        const unsigned int tax_lim_cons   ///< Does a constraint on tax caps exist or no limit?
) const noexcept
/**
 * Makes the leader level constraints for a country.
 * The constraints added are as follows:
 * @f{eqnarray}{
 *  t_i^{I} &\leq& \bar{t_i^{I}}\\
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
    if (tax_lim_cons) {
        for (unsigned int follower = 0; follower < Params.n_followers; follower++) {
            // Constraints for Tax limits
            LeadCons(follower, Loc.at(Models::LeaderVars::Tax) + follower) = 1;
            LeadRHS(follower) = Params.LeaderParam.max_tax;
        }
    }
    // Export - import <= Local Production
    // (28b)
    for (unsigned int i = 0; i < Params.n_followers; i++)
        LeadCons.at(Params.n_followers, i) = -1;
    LeadCons.at(Params.n_followers, Loc.at(Models::LeaderVars::NetExport)) = 1;
    LeadCons.at(Params.n_followers, Loc.at(Models::LeaderVars::NetImport)) = -1;
    // Import limit - In more precise terms, everything that comes in minus everything that goes out should satisfy this limit
    // (28c)
    if (import_lim_cons) {
        LeadCons(Params.n_followers + import_lim_cons, Loc.at(Models::LeaderVars::NetImport)) = 1;
        LeadCons(Params.n_followers + import_lim_cons, Loc.at(Models::LeaderVars::NetExport)) = -1;
        LeadRHS(Params.n_followers + import_lim_cons) = Params.LeaderParam.import_limit;
    }
    // Export limit - In more precise terms, everything that goes out minus everything that comes in should satisfy this limit
    // (28d)
    if (export_lim_cons) {
        LeadCons(Params.n_followers + import_lim_cons + export_lim_cons, Loc.at(Models::LeaderVars::NetExport)) = 1;
        LeadCons(Params.n_followers + import_lim_cons + export_lim_cons, Loc.at(Models::LeaderVars::NetImport)) = -1;
        LeadRHS(Params.n_followers + import_lim_cons + export_lim_cons) = Params.LeaderParam.export_limit;
    }
    // (28g)
    if (price_lim_cons) {
        for (unsigned int i = 0; i < Params.n_followers; i++)
            LeadCons.at(Params.n_followers + price_lim_cons + import_lim_cons + export_lim_cons,
                        i) = -Params.DemandParam.beta;
        LeadCons.at(Params.n_followers + price_lim_cons + import_lim_cons + export_lim_cons,
                    Loc.at(Models::LeaderVars::NetImport)) = -Params.DemandParam.beta;
        LeadCons.at(Params.n_followers + price_lim_cons + import_lim_cons + export_lim_cons,
                    Loc.at(Models::LeaderVars::NetExport)) = Params.DemandParam.beta;
        LeadRHS.at(Params.n_followers + price_lim_cons + import_lim_cons + export_lim_cons) =
                Params.LeaderParam.price_limit - Params.DemandParam.alpha;
    }
    if (VERBOSE) {
        cout << "\n********** Price Limit constraint: " << price_lim_cons;
        cout << "\n********** Import Limit constraint: " << import_lim_cons;
        cout << "\n********** Export Limit constraint: " << export_lim_cons;
        cout << "\n********** Tax Limit constraint: " << tax_lim_cons << "\n\t";
        for (unsigned int i = 0; i < Params.n_followers; i++) cout << "q_" + to_string(i) << "\t\t";
        cout << "q_imp\t\tq_exp\t\tp_cap\t\t";
        for (unsigned int i = 0; i < Params.n_followers; i++) cout << "t_" + to_string(i) << "\t\t";
        LeadCons.impl_print_dense("\nLeadCons:\n");
        LeadRHS.print("\nLeadRHS");
    }
}


Models::EPEC &
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
 * Similarly the leader cannot also impose tax on any player greater than what is dictated by the input variable `max_tax`.
 *
 * @return Pointer to LCP object dynamically created using `new`.
 */
{
    if (this->finalized)
        throw string(
                "Error in Models::EPEC::addCountry: EPEC object finalized. Call EPEC::unlock() to unlock this object first and then edit.");

    bool noError = false;
    try { noError = this->ParamValid(Params); }
    catch (const char *e) { cerr << "Error in Models::EPEC::addCountry: " << e << endl; }
    catch (string e) { cerr << "String: Error in Models::EPEC::addCountry: " << e << endl; }
    catch (exception &e) { cerr << "Exception: Error in Models::EPEC::addCountry: " << e.what() << endl; }
    if (!noError) return *this;

    const unsigned int LeadVars = 2 + 2 * Params.n_followers +
                                  addnlLeadVars;// two for quantity imported and exported, n for imposed cap and last n for tax

    LeadLocs Loc;
    Models::init(Loc);

    // Allocate so much space for each of these types of variables
    Models::increaseVal(Loc, LeaderVars::FollowerStart, Params.n_followers);
    Models::increaseVal(Loc, LeaderVars::NetImport, 1);
    Models::increaseVal(Loc, LeaderVars::NetExport, 1);
    Models::increaseVal(Loc, LeaderVars::Caps, Params.n_followers);
    Models::increaseVal(Loc, LeaderVars::Tax, Params.n_followers);

    if (VERBOSE) cout << Loc.at(LeaderVars::Tax) << endl << " Country tax position above \n";

    // Loc[Models::LeaderVars::AddnVar] = 1;

    // Leader Constraints
    short int import_lim_cons{0}, export_lim_cons{0}, price_lim_cons{0}, tax_lim_cons{0};
    if (Params.LeaderParam.import_limit >= 0) import_lim_cons = 1;
    if (Params.LeaderParam.export_limit >= 0) export_lim_cons = 1;
    if (Params.LeaderParam.price_limit >= 0) price_lim_cons = 1;
    if (Params.LeaderParam.max_tax >= 0) tax_lim_cons = 1;

    // cout<<" In addCountry: "<<Loc[Models::LeaderVars::End]<<endl;
    arma::sp_mat LeadCons(import_lim_cons +    // Import limit constraint
                          export_lim_cons +                // Export limit constraint
                          price_lim_cons +                    // Price limit constraint
                          tax_lim_cons * Params.n_followers +                // Tax limit constraint
                          1,                                // Export - import <= Domestic production
                          Loc[Models::LeaderVars::End]
    );
    arma::vec LeadRHS(import_lim_cons +
                      export_lim_cons +
                      price_lim_cons +
                      tax_lim_cons * Params.n_followers +
                      1, arma::fill::zeros);

    vector<shared_ptr<Game::QP_Param>> Players{};
    // Create the QP_Param* for each follower
    try {
        for (unsigned int follower = 0; follower < Params.n_followers; follower++) {
            auto Foll = make_shared<Game::QP_Param>(this->env);
            this->make_LL_QP(Params, follower, Foll.get(), Loc);
            Players.push_back(Foll);
        }
    }
    catch (const char *e) {
        cerr << e << endl;
        throw;
    }
    catch (string e) {
        cerr << "String in Models::EPEC::addCountry : " << e << endl;
        throw;
    }
    catch (GRBException &e) {
        cerr << "GRBException in Models::EPEC::addCountry : " << e.getErrorCode() << ": " << e.getMessage() << endl;
        throw;
    }
    catch (exception &e) {
        cerr << "Exception in Models::EPEC::addCountry : " << e.what() << endl;
        throw;
    }

    // Make Leader Constraints
    try {
        this->make_LL_LeadCons(LeadCons, LeadRHS, Params, Loc, import_lim_cons, export_lim_cons, price_lim_cons,
                               tax_lim_cons);
    }
    catch (const char *e) {
        cerr << e << endl;
        throw;
    }
    catch (string e) {
        cerr << "String: " << e << endl;
        throw;
    }
    catch (GRBException &e) {
        cerr << "GRBException: " << e.getErrorCode() << ": " << e.getMessage() << endl;
        throw;
    }
    catch (exception &e) {
        cerr << "Exception in Models::EPEC::addCountry : " << e.what() << endl;
        throw;
    }

    // Lower level Market clearing constraints - empty
    arma::sp_mat MC(0, LeadVars + Params.n_followers);
    arma::vec MCRHS(0, arma::fill::zeros);

    //Convert the country QP to a NashGame
    auto N = std::make_shared<Game::NashGame>(Players, MC, MCRHS, LeadVars, LeadCons, LeadRHS);
    this->name2nos[Params.name] = this->countries_LL.size();
    this->countries_LL.push_back(N);
    Models::increaseVal(Loc, Models::LeaderVars::DualVar, N->getNduals());
    Locations.push_back(Loc);
    this->LeadConses.push_back(N->RewriteLeadCons());
    // cout<<LeadCons<<N->RewriteLeadCons()<<LeadConses.back();
    this->AllLeadPars.push_back(Params);
    nCountr++;
    return *this;
}

Models::EPEC &
Models::EPEC::addTranspCosts(const arma::sp_mat &costs ///< The transportation cost matrix
)
/**
 * @brief Adds intercountry transportation costs matrix
 * @details Adds the transportation cost matrix. Entry in row i and column j of this matrix corresponds to the unit transportation costs for sending fuel from country i to country j.
 */
{
    if (this->finalized)
        throw string(
                "Error in Models::EPEC::addTranspCosts: EPEC object finalized. Call EPEC::unlock() to unlock this object first and then edit.");
    try {
        if (this->nCountries != costs.n_rows || this->nCountries != costs.n_cols)
            throw "Error in EPEC::addTranspCosts. Invalid size of Q";
        else this->TranspCosts = arma::sp_mat(costs);
        this->TranspCosts.diag().zeros();        // Doesn't make sense for it to have a nonzero diagonal!
    }
    catch (const char *e) {
        cerr << e << endl;
        throw;
    }
    catch (string e) {
        cerr << "String in Models::EPEC::addTranspCosts : " << e << endl;
        throw;
    }
    catch (GRBException &e) {
        cerr << "GRBException in Models::EPEC::addTranspCosts : " << e.getErrorCode() << ": " << e.getMessage() << endl;
        throw;
    }
    catch (exception &e) {
        cerr << "Exception in Models::EPEC::addTranspCosts : " << e.what() << endl;
        throw;
    }

    return *this;
}

const
Models::EPEC &
Models::EPEC::
finalize()
/**
 * @brief Finalizes the creation of a Models::EPEC object.
 * @details Performs a bunch of job after all data for a Models::EPEC object are given, namely.
 * 	-	Adds the tradebalance constraints for leaders. Calls Models::EPEC::add_Leaders_tradebalance_constraints
 * 	-	Computes the location of where each Leader's variable start in the variable list. Calls Models::EPEC::computeLeaderLocations
 * 	-	Adds the required dummy variables to each leader's problem so that a game among the leaders can be defined. Calls Models::EPEC::add_Dummy_Lead
 * 	-	Makes the market clearing constraint in each country. Calls Models::EPEC::make_MC_leader
 * 	-	Creates the QP objective corresponding to each leader's objective. Calls Models::EPEC::make_obj_leader
 */
{
    if (this->finalized) cerr << "Warning in Models::EPEC::finalize: Model already finalized\n";
    try {
        /*
         * Below for loop adds space for each country's quantity imported from variable
         */
        this->nImportMarkets = vector<unsigned int>(this->nCountries);
        for (unsigned int i = 0; i < this->nCountries; i++)
            this->add_Leaders_tradebalance_constraints(i);

        /*
         * Now we keep track of where each country's variables start
         */
        this->computeLeaderLocations(true);

        this->MC_QP = vector<shared_ptr<Game::QP_Param>>(nCountries);
        this->LeadObjec = vector<shared_ptr<Game::QP_objective>>(nCountries);
        this->country_QP = vector<shared_ptr<Game::QP_Param>>(nCountries);
        for (unsigned int i = 0; i < this->nCountries; i++) // To add the corresponding Market Clearing constraint
        {
            Game::QP_objective QP_obj;
            this->add_Dummy_Lead(i);
            this->make_MC_leader(i);
            this->LeadObjec.at(i) = std::make_shared<Game::QP_objective>();
            this->make_obj_leader(i, *this->LeadObjec.at(i).get());
        }
    }
    catch (const char *e) {
        cerr << e << endl;
        throw;
    }
    catch (string e) {
        cerr << "String in Models::EPEC::finalize : " << e << endl;
        throw;
    }
    catch (GRBException &e) {
        cerr << "GRBException in Models::EPEC::finalize : " << e.getErrorCode() << ": " << e.getMessage() << endl;
        throw;
    }
    catch (exception &e) {
        cerr << "Exception in Models::EPEC::finalize : " << e.what() << endl;
        throw;
    }
    this->finalized = true;
    return *this;
}

void
Models::EPEC::add_Leaders_tradebalance_constraints(const unsigned int i)
/**
 * @brief Adds leaders' trade balance constraints for import-exports
 * @details Does the following job:
 * 	-	Counts the number of import markets for the country @p i to store in Models::EPEC::nImportMarkets
 * 	-	Adds the trade balance constraint. Total quantity imported by country @p i = Sum of Total quantity exported by each country to country i.
 * 	-	Updates the LeadLocs in Models::EPEC::Locations.at(i)
 */
{
    if (i >= this->nCountries)
        throw string("Error in Models::EPEC::add_Leaders_tradebalance_constraints. Bad argument");
    int nImp = 0;
    LeadLocs &Loc = this->Locations.at(i);
    // Counts the number of countries from which the current country imports
    for (auto val = TranspCosts.begin_col(i); val != TranspCosts.end_col(i); ++val) nImp++;
    // substitutes that answer to nImportMarkets at the current position
    this->nImportMarkets.at(i) = (nImp);
    if (nImp > 0) {
        Models::increaseVal(Loc, LeaderVars::CountryImport, nImp);

        Game::NashGame &LL_Nash = *this->countries_LL.at(i).get();

        // Adding the constraint that the sum of imports from all countries equals total imports
        arma::vec a(Loc.at(Models::LeaderVars::End) - LL_Nash.getNduals(), arma::fill::zeros);
        a.at(Loc.at(Models::LeaderVars::NetImport)) = -1;
        a.subvec(Loc.at(LeaderVars::CountryImport), Loc.at(LeaderVars::CountryImport + 1) - 1).ones();
        if (VERBOSE) {
            cout << endl << " ______ " << endl;
            for (auto v:Loc) cout << v.first << "\t\t\t" << v.second << endl;
            cout << endl << " ______ " << endl;
        }

        LL_Nash.addDummy(nImp, Loc.at(Models::LeaderVars::CountryImport));
        LL_Nash.addLeadCons(a, 0).addLeadCons(-a, 0);
    } else {
        Game::NashGame &LL_Nash = *this->countries_LL.at(i).get();

        // Set imports and exporta to zero
        arma::vec a(Loc.at(Models::LeaderVars::End) - LL_Nash.getNduals(), arma::fill::zeros);
        a.at(Loc.at(Models::LeaderVars::NetImport)) = 1;
        if (VERBOSE)
            cout << "Single Country: imports are set to zero." << endl;
        LL_Nash.addLeadCons(a, 0);
        a.at(Loc.at(Models::LeaderVars::NetImport)) = 0;
        a.at(Loc.at(Models::LeaderVars::NetExport)) = 1;
        if (VERBOSE)
            cout << "Single Country: exports are set to zero." << endl;
        LL_Nash.addLeadCons(a, 0);
    }
    // Updating the variable locations
    /*	Loc[Models::LeaderVars::CountryImport] = Loc.at(Models::LeaderVars::End);
       Loc.at(Models::LeaderVars::End) += nImp;*/
}

void
Models::EPEC::make_MC_cons(arma::sp_mat &MCLHS, arma::vec &MCRHS) const
/** @brief Returns leader's Market clearing constraints in matrix form
 * @details
 */
{
    if (!this->finalized)
        throw string("Error in Models::EPEC::make_MC_cons: This function can be run only AFTER calling finalize()");
    // Transportation matrix
    const arma::sp_mat &TrCo = this->TranspCosts;
    // Output matrices
    MCRHS.zeros(this->nCountries);
    MCLHS.zeros(this->nCountries, this->nVarEPEC);
    // The MC constraint for each leader country
    if (this->nCountries > 1) {
        for (unsigned int i = 0; i < this->nCountries; ++i) {
            MCLHS(i, this->getPosition(i, LeaderVars::NetExport)) = 1;
            for (auto val = TrCo.begin_row(i); val != TrCo.end_row(i); ++val) {
                const unsigned int j = val.col(); // This is the country which is importing from "i"
                unsigned int count{0};

                for (auto val2 = TrCo.begin_col(j); val2 != TrCo.end_col(j); ++val2)
                    // What position in the list of j's impoting from countries  does i fall in?
                {
                    if (val2.row() == i) break;
                    else count++;
                }
                MCLHS(i,
                      this->getPosition(j, Models::LeaderVars::CountryImport) + count
                ) = -1;
            }
        }
    }
}

void
Models::EPEC::make_MC_leader(const unsigned int i)
/**
 * @brief Makes the market clearing constraint for country @p i
 * @details Writes the market clearing constraint as a Game::QP_Param and stores it in Models::EPEC::MC_QP
 */
{
    if (i >= this->nCountries)
        throw string("Error in Models::EPEC::add_Leaders_tradebalance_constraints. Bad argument");
    try {
        const arma::sp_mat &TrCo = this->TranspCosts;
        const unsigned int nEPECvars = this->nVarEPEC;
        const unsigned int nThisMCvars = 1;
        arma::sp_mat C(nThisMCvars, nEPECvars - nThisMCvars);


        C.at(0, this->getPosition(i, Models::LeaderVars::NetExport)) = 1;

        for (auto val = TrCo.begin_row(i); val != TrCo.end_row(i); ++val) {
            const unsigned int j = val.col(); // This is the country which the country "i" is importing from
            unsigned int count{0};

            for (auto val2 = TrCo.begin_col(j); val2 != TrCo.end_col(j); ++val2)
                // What position in the list of j's impoting from countries  does i fall in?
            {
                if (val2.row() == i) break;
                else count++;
            }

            C.at(0, this->getPosition(j,
                                      Models::LeaderVars::CountryImport) + count
                    - (j >= i ? nThisMCvars : 0)) = 1;
        }

        this->MC_QP.at(i) = std::make_shared<Game::QP_Param>(this->env);
        // Note Q = {{0}}, c={0}, the MC problem has no constraints. So A=B={{}}, b={}.
        this->MC_QP.at(i).get()->set(
                arma::sp_mat{1, 1},                        // Q
                std::move(C),                            // C
                arma::sp_mat{0, nEPECvars - nThisMCvars},    // A
                arma::sp_mat{0, nThisMCvars},            // B
                arma::vec{0},                            // c
                arma::vec{}                                // b
        );
    }
    catch (const char *e) {
        cerr << e << endl;
        throw;
    }
    catch (string e) {
        cerr << "String in Models::EPEC::make_MC_leader : " << e << endl;
        throw;
    }
    catch (GRBException &e) {
        cerr << "GRBException in Models::EPEC::make_MC_leader : " << e.getErrorCode() << ": " << e.getMessage() << endl;
        throw;
    }
    catch (exception &e) {
        cerr << "Exception in Models::EPEC::make_MC_leader : " << e.what() << endl;
        throw;
    }
}

bool
Models::EPEC::dataCheck(
        const bool chkAllLeadPars,        ///< Checks if Models::EPEC::AllLeadPars has size @p n
        const bool chkcountries_LL,        ///< Checks if Models::EPEC::countries_LL has size @p n
        const bool chkMC_QP,            ///< Checks if Models::EPEC::MC_QP has size @p n
        const bool chkLeadConses,        ///< Checks if Models::EPEC::LeadConses has size @p n
        const bool chkLeadRHSes,        ///< Checks if Models::EPEC::LeadRHSes has size @p n
        const bool chknImportMarkets,    ///< Checks if Models::EPEC::nImportMarkets has size @p n
        const bool chkLocations,        ///< Checks if Models::EPEC::Locations has size @p n
        const bool chkLeaderLocations,    ///< Checks if Models::EPEC::LeaderLocations has size @p n and Models::EPEC::nVarEPEC is set
        const bool chkLeadObjec            ///< Checks if Models::EPEC::LeadObjec has size @p n
) const
/**
 * Checks the data in Models::EPEC object, based on checking flags, @p n is the number of countries in the Models::EPEC object.
 */
{
    if (!chkAllLeadPars && AllLeadPars.size() != this->nCountries) return false;
    if (!chkcountries_LL && countries_LL.size() != this->nCountries) return false;
    if (!chkMC_QP && MC_QP.size() != this->nCountries) return false;
    if (!chkLeadConses && LeadConses.size() != this->nCountries) return false;
    if (!chkLeadRHSes && LeadRHSes.size() != this->nCountries) return false;
    if (!chknImportMarkets && nImportMarkets.size() != this->nCountries) return false;
    if (!chkLocations && Locations.size() != this->nCountries) return false;
    if (!chkLeaderLocations && LeaderLocations.size() != this->nCountries) return false;
    if (!chkLeaderLocations && this->nVarEPEC == 0) return false;
    if (!chkLeadObjec && LeadObjec.size() != this->nCountries) return false;
    return true;
}

void
Models::EPEC::add_Dummy_Lead(const unsigned int i) {
    if (!this->dataCheck()) throw string("Error in Models::EPEC::add_Dummy_All_Lead: dataCheck() failed!");

    const unsigned int nEPECvars = this->nVarEPEC;
    const unsigned int nThisCountryvars = this->Locations.at(i).at(Models::LeaderVars::End);

    try {
        this->countries_LL.at(i).get()->addDummy(nEPECvars - nThisCountryvars);
    }
    catch (const char *e) {
        cerr << e << endl;
        throw;
    }
    catch (string e) {
        cerr << "String in Models::EPEC::add_Dummy_All_Lead : " << e << endl;
        throw;
    }
    catch (GRBException &e) {
        cerr << "GRBException in Models::EPEC::add_Dummy_All_Lead : " << e.getErrorCode() << ": " << e.getMessage()
             << endl;
        throw;
    }
    catch (exception &e) {
        cerr << "Exception in Models::EPEC::add_Dummy_All_Lead : " << e.what() << endl;
        throw;
    }
}

void
Models::EPEC::computeLeaderLocations(const bool addSpaceForMC) {
    this->LeaderLocations = vector<unsigned int>(this->nCountries);
    this->LeaderLocations.at(0) = 0;
    for (unsigned int i = 1; i < this->nCountries; i++)
        this->LeaderLocations.at(i) = this->getPosition(i - 1, Models::LeaderVars::End) + (addSpaceForMC ? 0 : 0);

    this->nVarinEPEC =
            this->getPosition(this->nCountries - 1, Models::LeaderVars::End) + (addSpaceForMC ? this->nCountries : 0);
}

unsigned int
Models::EPEC::getPosition(const unsigned int countryCount, const Models::LeaderVars var) const
/**
 * @brief Gets position of a variable in a country.
 */
{
    if (countryCount > this->nCountries) throw string("Error in Models::EPEC::getPosition: Bad Country Count");
    return this->LeaderLocations.at(countryCount) + this->Locations.at(countryCount).at(var);
}

unsigned int
Models::EPEC::getPosition(const string countryName, const Models::LeaderVars var) const
/**
 * @brief Gets position of a variable in a country given the country name and the variable.
 */
{
    return this->getPosition(name2nos.at(countryName), var);
}

Game::NashGame *
Models::EPEC::get_LowerLevelNash(const unsigned int i) const
/**
 * @brief Returns a non-owning pointer to the @p i -th country's lower level NashGame
 */
{
    return this->countries_LL.at(i).get();
}

Models::EPEC &
Models::EPEC::unlock()
/**
 * @brief Unlocks an EPEC model
 * @details A finalized model cannot be edited unless it is unlocked first.
 * @internal EPEC::finalize() performs "finalizing" acts on an object.
 * @warning Exclusively for debugging purposes for developers. Don't call this function, unless you know what you are doing.
 */
{
    this->finalized = false;
    return *this;
}


void
Models::EPEC::make_obj_leader(const unsigned int i, ///< The location of the country whose objective is to be made
                              Game::QP_objective &QP_obj                    ///< The object where the objective parameters are to be stored.
)
/**
 * Makes the objective function of each country.
 */
{
    const unsigned int nEPECvars = this->nVarEPEC;
    const unsigned int nThisCountryvars = this->Locations.at(i).at(Models::LeaderVars::End);
    const LeadAllPar &Params = this->AllLeadPars.at(i);
    const arma::sp_mat &TrCo = this->TranspCosts;
    const LeadLocs &Loc = this->Locations.at(i);

    QP_obj.Q.zeros(nEPECvars - nThisCountryvars, nEPECvars - nThisCountryvars);
    QP_obj.c.zeros(nThisCountryvars);
    QP_obj.C.zeros(nThisCountryvars, nEPECvars - nThisCountryvars);
    // emission term
    for (unsigned int j = Loc.at(Models::LeaderVars::FollowerStart), count = 0;
         count < Params.n_followers;
         j++, count++)
        QP_obj.c.at(j) = Params.FollowerParam.emission_costs.at(count);
    if (this->nCountries > 1) {
        // export revenue term
        QP_obj.C(Loc.at(Models::LeaderVars::NetExport),
                 this->getPosition(i, Models::LeaderVars::End) - nThisCountryvars) = -1;
        // Import cost term.
        unsigned int count{0};
        for (auto val = TrCo.begin_col(i); val != TrCo.end_col(i); ++val, ++count) {
            // C^{tr}_{IA}*q^{I\to A}_{imp} term
            QP_obj.c.at(Loc.at(Models::LeaderVars::CountryImport) + count) = (*val);
            // \pi^I*q^{I\to A}_{imp} term
            QP_obj.C.at(Loc.at(Models::LeaderVars::CountryImport) + count,
                        this->getPosition(val.row(), Models::LeaderVars::End)) = 1;
        }
    }
}

unique_ptr<GRBModel>
Models::EPEC::Respond(const string name, const arma::vec &x) const {
    return this->Respond(this->name2nos.at(name), x);
}

unique_ptr<GRBModel>
Models::EPEC::Respond(const unsigned int i, const arma::vec &x) const {
    if (!this->finalized) throw string("Error in Models::EPEC::Respond: Model not finalized");

    if (i >= this->nCountries) throw string("Error in Models::EPEC::Respond: Invalid country number");

    const unsigned int nEPECvars = this->nVarEPEC;
    const unsigned int nThisCountryvars = this->Locations.at(i).at(Models::LeaderVars::End);

    if (x.n_rows != nEPECvars - nThisCountryvars)
        throw string("Error in Models::EPEC::Respond: Invalid parametrization");

    return this->country_QP.at(i).get()->solveFixed(x);
}

void
Models::EPEC::make_country_QP()
/**
 * @brief Makes the Game::QP_Param for all the countries
 * @details
 * Calls are made to Models::EPEC::make_country_QP(const unsigned int i) for each valid @p i
 * @note Overloaded as EPEC::make_country_QP(unsigned int)
 */
{
    static bool already_ran{false};
    if (!already_ran)
        for (unsigned int i = 0; i < this->nCountries; ++i)
            this->make_country_QP(i);
    for (unsigned int i = 0; i < this->nCountries; ++i) {
        LeadLocs &Loc = this->Locations.at(i);
        // Adjusting "stuff" because we now have new convHull variables
        unsigned int convHullVarCount = this->LeadObjec.at(i)->Q.n_rows - Loc[Models::LeaderVars::End];
        // Location details
        Models::increaseVal(Loc, Models::LeaderVars::ConvHullDummy, convHullVarCount);
        // All other players' QP
        if (this->nCountries > 1) {
            for (unsigned int j = 0; j < this->nCountries; j++) {
                if (i != j)
                    this->country_QP.at(j)->addDummy(convHullVarCount, 0);
                this->MC_QP.at(j)->addDummy(convHullVarCount, 0);
            }
        }
    }
    this->computeLeaderLocations(true);
    if (VERBOSE) {
        for (unsigned int i = 0; i < this->nCountries; ++i)
            this->country_QP.at(i)->QP_Param::write("dat/countrQP_" + to_string(i), false);
    }
}

void
Models::EPEC::make_country_QP(const unsigned int i)
/**
 * @brief Makes the Game::QP_Param corresponding to the @p i-th country.
 * @details
 *  - First gets the Game::LCP object from @p countries_LL and makes a QP with this LCP as the lower level
 *  - This is achieved by calling LCP::makeQP and using the objective value object in @p LeadObjec
 *  - Finally the locations are updated owing to the complete convex hull calculated during the call to LCP::makeQP
 * @note Overloaded as EPEC::make_country_QP()
 * @todo where is the error?
 */
{
    if (!this->finalized) throw string("Error in Models::EPEC::make_country_QP: Model not finalized");
    if (i >= this->nCountries) throw string("Error in Models::EPEC::make_country_QP: Invalid country number");
    if (!this->country_QP.at(i).get()) {
        Game::LCP Player_i_LCP = Game::LCP(this->env, *this->countries_LL.at(i).get());
        if (VERBOSE) cout << "In EPEC::make_country_QP: " << Player_i_LCP.getCompl().size() << endl;
        this->country_QP.at(i) = std::make_shared<Game::QP_Param>(this->env);
        Player_i_LCP.makeQP(*this->LeadObjec.at(i).get(), *this->country_QP.at(i).get());
    }
}


void
Models::increaseVal(LeadLocs &L, const LeaderVars start, const unsigned int val, const bool startnext)
/**
 * Should be called ONLY after initializing @p L by calling Models::init
 */
{
    LeaderVars start_rl = startnext ? start + 1 : start;
    for (LeaderVars l = start_rl; l != Models::LeaderVars::End; l = l + 1)
        L[l] += val;
    L[Models::LeaderVars::End] += val;
}


void
Models::init(LeadLocs &L) {
    for (LeaderVars l = Models::LeaderVars::FollowerStart; l != Models::LeaderVars::End; l = l + 1) L[l] = 0;
    L[Models::LeaderVars::End] = 0;
}

Models::LeaderVars Models::operator+(Models::LeaderVars a, int b) {
    return static_cast<LeaderVars>(static_cast<int> (a) + b);
}

void
Models::EPEC::findNashEq(bool write, string filename) {
    if (this->country_QP.front() != nullptr) {

        int Nvar = this->country_QP.front()->getNx() + this->country_QP.front()->getNy();
        arma::sp_mat MC(0, Nvar), dumA(0, Nvar);
        arma::vec MCRHS, dumb;
        MCRHS.zeros(0);
        dumb.zeros(0);
        this->make_MC_cons(MC, MCRHS);
        this->nashgame = std::unique_ptr<Game::NashGame>(
                new Game::NashGame(this->country_QP, MC, MCRHS, 0, dumA, dumb));
        //if (VERBOSE) cout << *nashgame << endl;
        lcp = std::unique_ptr<Game::LCP>(new Game::LCP(this->env, *nashgame));

        if (VERBOSE) this->nashgame->write("dat/NashGame", false, true);
        //Using indicator constraints
        lcp->useIndicators = this->indicators;
        this->lcpmodel = lcp->LCPasMIP(false);

        Nvar = nashgame->getNprimals() + nashgame->getNduals() + nashgame->getNshadow() + nashgame->getNleaderVars();
        lcpmodel->optimize();
        if (VERBOSE) {
            lcpmodel->write("dat/NashLCP.lp");
            lcpmodel->write("dat/NashLCP.sol");
        }
        this->sol_x.zeros(Nvar);
        this->sol_z.zeros(Nvar);
        unsigned int temp;
        int status = lcpmodel->get(GRB_IntAttr_Status);
        if (status != GRB_INF_OR_UNBD && status != GRB_INFEASIBLE && status != GRB_INFEASIBLE) {
            try {

                for (unsigned int i = 0; i < (unsigned int) Nvar; i++) {
                    this->sol_x(i) = lcpmodel->getVarByName("x_" + to_string(i)).get(GRB_DoubleAttr_X);
                    this->sol_z(i) = lcpmodel->getVarByName("z_" + to_string(i)).get(GRB_DoubleAttr_X);
                    //if (VERBOSE)
                    //    cout << "x_" + to_string(i) + ":" << this->sol_x(i) << "\t\tz_" + to_string(i) + ":"
                    //         << this->sol_z(i) << endl;
                    temp = i;
                }

            }
            catch (GRBException &e) {
                cerr << "GRBException in Models::EPEC::findNashEq : " << e.getErrorCode() << ": " << e.getMessage()
                     << " "
                     << temp << endl;
            }
            if (write) {
                this->sol_x.save("dat/x_" + filename, arma::file_type::arma_ascii, VERBOSE);
                this->sol_z.save("dat/z_" + filename, arma::file_type::arma_ascii, VERBOSE);
                try {
                    this->WriteCountry(0, "dat/Solution.txt", this->sol_x, false);
                    for (unsigned int ell = 1; ell < this->nCountries; ++ell)
                        this->WriteCountry(ell, "dat/Solution.txt", this->sol_x, true);
                    this->write("dat/Solution.txt", true);
                } catch (GRBException &e) {}
            }
        } else
            cout << "Models::EPEC::findNashEq: no nash equilibrium found." << endl;
        //if (VERBOSE) Game::print(lcp->getCompl());

    } else {
        cerr << "GRBException in Models::EPEC::findNashEq : no country QP has been made." << endl;
        throw;
    }
}


void Models::EPEC::gur_WriteCountry_conv(const unsigned int i, string filename) const {

    if (!lcp) throw;
}

void Models::EPEC::gur_WriteEpecMip(const unsigned int i, string filename) const {

    if (!lcp) throw;
}

string to_string(const GRBConstr &cons, const GRBModel &model) {
    const GRBVar *vars = model.getVars();
    const int nVars = model.get(GRB_IntAttr_NumVars);
    ostringstream oss;
    oss << cons.get(GRB_StringAttr_ConstrName) << ":\t\t";
    constexpr double eps = 1e-5;
    // LHS
    for (int i = 0; i < nVars; ++i) {
        double coeff = model.getCoeff(cons, vars[i]);
        if (abs(coeff) > eps) {
            char sign = (coeff > eps) ? '+' : ' ';
            oss << sign << coeff << to_string(vars[i]) << "\t";
        }
    }
    // Inequality/Equality and RHS
    oss << cons.get(GRB_CharAttr_Sense) << "\t" << cons.get(GRB_DoubleAttr_RHS);
    return oss.str();
}

string to_string(const GRBVar &var) {
    string name = var.get(GRB_StringAttr_VarName);
    return name.empty() ? "unNamedvar" : name;
}


void
Models::EPEC::write(const string filename, const unsigned int i, bool append) const {
    ofstream file;
    file.open(filename, append ? ios::app : ios::out);
    const LeadAllPar &Params = this->AllLeadPars.at(i);
    file << "**************************************************\n";
    file << "COUNTRY: " << Params.name << '\n';
    file << "- - - - - - - - - - - - - - - - - - - - - - - - - \n";
    file << Params;
    file << "**************************************************\n\n\n\n\n";
    file.close();
}


void
Models::EPEC::write(const string filename, bool append) const {
    if (append) {
        ofstream file;
        file.open(filename, ios::app);
        file << "\n\n\n\n\n";
        file << "##################################################\n";
        file << "############### COUNTRY PARAMETERS ###############\n";
        file << "##################################################\n";
    }
    for (unsigned int i = 0; i < this->nCountries; ++i)
        this->write(filename, i, (append || i));
}


void
Models::EPEC::WriteCountry(const unsigned int i, const string filename, const arma::vec x, const bool append) const {
    //if (!lcp) return;
    // const LeadLocs& Loc = this->Locations.at(i);

    ofstream file;
    file.open(filename, append ? ios::app : ios::out);
    // FILE OPERATIONS START
    const LeadAllPar &Params = this->AllLeadPars.at(i);
    file << "**************************************************\n";
    file << "COUNTRY: " << Params.name << '\n';
    file << "**************************************************\n\n";
    // Country Variables
    unsigned int foll_prod;
    foll_prod = this->getPosition(i, Models::LeaderVars::FollowerStart);
    // Domestic production
    double prod{0};
    for (unsigned int j = 0; j < Params.n_followers; ++j) prod += x.at(foll_prod + j);
    // Trade
    double Export{x.at(this->getPosition(i, Models::LeaderVars::NetExport))};
    double import{0};
    for (unsigned int j = this->getPosition(i, Models::LeaderVars::CountryImport);
         j < this->getPosition(i, Models::LeaderVars::CountryImport + 1); ++j)
        import += x.at(j);
    // Writing national level details
    file << Models::prn::label << "Domestic production" << ":" << Models::prn::val << prod << "\n";
    if (Export >= import)
        file << Models::prn::label << "Net exports" << ":" << Models::prn::val << Export - import << "\n";
    else
        file << Models::prn::label << "Net imports" << ":" << Models::prn::val << import - Export << "\n";
    file << Models::prn::label << " -> Total Export" << ":" << Models::prn::val << Export << "\n";
    file << Models::prn::label << " -> Total Import" << ":" << Models::prn::val << import << endl;
    file << Models::prn::label << "Domestic consumed quantity" << ":" << Models::prn::val << import - Export + prod
         << "\n";
    file << Models::prn::label << "Domestic price" << ":" << Models::prn::val
         << Params.DemandParam.alpha - Params.DemandParam.beta * (import - Export + prod) << "\n";

    file.close();

    // Follower productions
    file << "- - - - - - - - - - - - - - - - - - - - - - - - - \n";
    file << "FOLLOWER DETAILS:\n";
    for (unsigned int j = 0; j < Params.n_followers; ++j)
        this->WriteFollower(i, j, filename, x);

    file << "\n\n\n";
    // FILE OPERATIONS END
}

void Models::EPEC::WriteFollower(const unsigned int i, const unsigned int j, const string filename,
                                 const arma::vec x) const {
    ofstream file;
    file.open(filename, ios::app);

    // Country Variables
    const LeadAllPar &Params = this->AllLeadPars.at(i);
    unsigned int foll_prod, foll_tax, foll_lim;
    foll_prod = this->getPosition(i, Models::LeaderVars::FollowerStart);
    foll_tax = this->getPosition(i, Models::LeaderVars::Tax);
    foll_lim = this->getPosition(i, Models::LeaderVars::Caps);

    string name;
    try { name = Params.name + " --- " + Params.FollowerParam.names.at(j); }
    catch (...) { name = "Follower " + to_string(j) + " of leader " + to_string(i); }

    file << "\n" << name << "\n\n";//<<" named "<<Params.FollowerParam.names.at(j)<<"\n";

    const double q = x.at(foll_prod + j);
    const double tax = x.at(foll_tax + j);
    const double lim = x.at(foll_lim + j);
    const double lin = Params.FollowerParam.costs_lin.at(j);
    const double quad = Params.FollowerParam.costs_quad.at(j);

    file << Models::prn::label << "Quantity produced" << ":" << Models::prn::val << q << endl;
    //file << "x(): " << foll_prod + j << endl;
    file << Models::prn::label << "Capacity of production" << ":" << Models::prn::val
         << Params.FollowerParam.capacities.at(j) << "\n";
    file << Models::prn::label << "Limit on production" << ":" << Models::prn::val << lim << "\n";
    //file << "x(): " << foll_lim + j << endl;
    file << Models::prn::label << "Tax imposed" << ":" << Models::prn::val << tax << "\n";
    //file << "x(): " << foll_tax + j << endl;
    file << Models::prn::label << "  -Production cost function" << ":" << "\t C(q) = (" << lin << " + " << tax
         << ")*q + 0.5*" << quad << "*q^2\n" << Models::prn::label << " " << "=" << Models::prn::val
         << (lin + tax) * q + 0.5 * quad * q * q << "\n";
    file << Models::prn::label << "  -Marginal cost of production" << ":" << Models::prn::val << quad * q + lin + tax
         << "\n";
    file << Models::prn::label << "Emission cost" << ":" << Models::prn::val
         << Params.FollowerParam.emission_costs.at(j) << endl;

    file.close();
}

void Models::EPEC::testQP(const unsigned int i) {
    QP_Param *QP = this->country_QP.at(i).get();
    arma::vec x;
    //if (VERBOSE) cout << *QP << endl;
    x.ones(QP->getNx());
    x.fill(555);
    if (VERBOSE) cout << "*** COUNTRY QP TEST***\n";
    std::unique_ptr<GRBModel> model = QP->solveFixed(x);
    if (VERBOSE) model->write("dat/CountryQP_" + to_string(i) + ".lp");
    int status = model->get(GRB_IntAttr_Status);
    if (status != GRB_INF_OR_UNBD && status != GRB_INFEASIBLE && status != GRB_INFEASIBLE) {
        arma::vec sol;
        sol.zeros(QP->getNy());
        try {
            GRBVar *vars = model->getVars();
            int i = 0;
            for (GRBVar *p = vars; i < model->get(GRB_IntAttr_NumVars); i++, p++) {
                sol.at(i) = p->get(GRB_DoubleAttr_X);
            }
        } catch (GRBException &e) {
            cerr << "GRBException in Models::EPEC::testQP: " << e.getErrorCode() << ": " << e.getMessage() << endl;
        }
        sol.save("dat/QP_Sol_" + to_string(i) + ".txt", arma::arma_ascii);
        this->WriteCountry(i, "dat/temp_" + to_string(i) + ".txt", sol, false);
    } else {
        cout << "Models::EPEC::testQP: QP of country " << i << " is infeasible or unbounded." << endl;
    }
}

void Models::EPEC::testLCP(const unsigned int i) {
    auto country = this->get_LowerLevelNash(i);
    LCP CountryLCP = LCP(this->env, *country);
    CountryLCP.write("dat/LCP_" + to_string(i));
    cout << "*** COUNTRY TEST***\n";
    auto model = CountryLCP.LCPasMIP(true);
    model->write("dat/CountryLCP_" + to_string(i) + ".lp");
    model->write("dat/CountryLCP_" + to_string(i) + ".sol");
}

