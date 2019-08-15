#ifndef MODELS_H
#define MODELS_H

#include"epecsolve.h"
#include<iostream>
#include<memory>
#include<gurobi_c++.h>
#include<armadillo>

namespace Models {
    typedef struct FollPar FollPar;
    typedef struct DemPar DemPar;
    typedef struct LeadPar LeadPar;
    typedef struct LeadAllPar LeadAllPar;

/// @brief Stores the parameters of the follower in a country model
    struct FollPar {
        vector<double> costs_quad = {};    ///< Quadratic coefficient of i-th follower's cost. Size of this vector should be equal to n_followers
        vector<double> costs_lin = {};    ///< Linear  coefficient of i-th follower's cost. Size of this vector should be equal to n_followers
        vector<double> capacities = {};    ///< Production capacity of each follower. Size of this vector should be equal to n_followers
        vector<double> emission_costs = {};    ///< Emission costs for unit quantity of the fuel. Emission costs feature only on the leader's problem
        vector<double> tax_caps = {};    ///< Individual tax caps for each follower.
        vector<string> names = {};    ///< Optional Names for the Followers.
        FollPar(vector<double> costs_quad_ = {}, vector<double> costs_lin_ = {}, vector<double> capacities_ = {},
                vector<double> emission_costs_ = {}, double tax_caps_ = {}, vector<string> names_ = {})
                : costs_quad{costs_quad_}, costs_lin{costs_lin_}, capacities{capacities_},
                  emission_costs{emission_costs_}, tax_caps(tax_caps_, costs_quad_.size()), names{names_} {}
    };


/// @brief Stores the parameters of the demand curve in a country model
    struct DemPar {
        double alpha = 100;    ///< Intercept of the demand curve. Written as: Price = alpha - beta*(Total quantity in domestic market)
        double beta = 2;    ///< Slope of the demand curve. Written as: Price = alpha - beta*(Total quantity in domestic market)
        DemPar(double alpha = 100, double beta = 2) : alpha{alpha}, beta{beta} {};
    };

/// @brief Stores the parameters of the leader in a country model
    struct LeadPar {
        double import_limit = -1;    ///< Maximum net import in the country. If no limit, set the value as -1;
        double export_limit = -1;    ///< Maximum net export in the country. If no limit, set the value as -1;
        double price_limit = -1;    ///< Government does not want the price to exceed this limit
        LeadPar(double imp_lim = -1, double exp_lim = -1, double price_limit = -1)
                : import_limit{imp_lim}, export_limit{exp_lim}, price_limit{price_limit} {}
    };

/// @brief Stores the parameters of a country model
    struct LeadAllPar {
        unsigned int n_followers;            ///< Number of followers in the country
        string name;                        ///< Country Name
        Models::FollPar FollowerParam = {};    ///< A struct to hold Follower Parameters
        Models::DemPar DemandParam = {};    ///< A struct to hold Demand Parameters
        Models::LeadPar LeaderParam = {};    ///< A struct to hold Leader Parameters
        LeadAllPar(unsigned int n_foll, string name, Models::FollPar FP = {}, Models::DemPar DP = {},
                   Models::LeadPar LP = {})
                : n_followers{n_foll}, name{name}, FollowerParam{FP}, DemandParam{DP}, LeaderParam{LP} {
            // Nothing here
        }
    };


    enum class LeaderVars {
        FollowerStart,
        NetImport,
        NetExport,
        CountryImport,
        Caps,
        Tax,
        DualVar,
        ConvHullDummy,
        End
    };
    ostream &operator<<(ostream &ost, const FollPar P);

    ostream &operator<<(ostream &ost, const DemPar P);

    ostream &operator<<(ostream &ost, const LeadPar P);

    ostream &operator<<(ostream &ost, const LeadAllPar P);

    ostream &operator<<(ostream &ost, const LeaderVars l);

    using LeadLocs=map<LeaderVars, unsigned int>;

    void increaseVal(LeadLocs &L, const LeaderVars start, const unsigned int val, const bool startnext = true);

    void init(LeadLocs &L);

    LeaderVars operator+(Models::LeaderVars a, int b);


    class EPEC {
    private:
        vector<LeadAllPar> AllLeadPars = {};  ///< The parameters of each leader in the EPEC game
        vector<shared_ptr<Game::NashGame>> countries_LL = {}; ///< Stores each country's lower level Nash game
        vector<shared_ptr<Game::QP_Param>> MC_QP = {};    ///< The QP corresponding to the market clearing condition of each player
        vector<shared_ptr<Game::QP_Param>> country_QP = {};    ///< The QP corresponding to each player
        vector<shared_ptr<Game::QP_objective>> LeadObjec = {};    ///< Objective of each leader
        vector<arma::sp_mat> LeadConses = {};        ///< Stores each country's leader constraint LHS
        vector<arma::vec> LeadRHSes = {};            ///< Stores each country's leader constraint RHS
        arma::sp_mat TranspCosts = {};                ///< Transportation costs between pairs of countries
        vector<unsigned int> nImportMarkets = {};    ///< Number of countries from which the i-th country imports
        vector<LeadLocs> Locations = {};            ///< Location of variables for each country
        vector<unsigned int> LeaderLocations = {};    ///< Location of each leader

        unique_ptr<Game::NashGame> nashgame;
        unique_ptr<Game::LCP> lcp;
        unique_ptr<GRBModel> lcpmodel;

        unsigned int nVarinEPEC{0};

        GRBEnv *env;        ///< A gurobi environment to create and process the resulting LCP object.
        map<string, unsigned int> name2nos = {};
        bool finalized{false}, nashEq{false};
        unsigned int nCountr = 0;

        bool dataCheck(const bool chkAllLeadPars = true, const bool chkcountriesLL = true, const bool chkMC_QP = true,
                       const bool chkLeadConses = true, const bool chkLeadRHSes = true,
                       const bool chknImportMarkets = true,
                       const bool chkLocations = true, const bool chkLeaderLocations = true,
                       const bool chkLeadObjec = true) const;

        // Super low level
        /// Checks that the parameter given to add a country is valid. Does not have obvious errors
        bool ParamValid(const LeadAllPar &Param) const;

        /// Makes the lower level quadratic program object for each follower.
        void make_LL_QP(const LeadAllPar &Params,
                        const unsigned int follower,
                        Game::QP_Param *Foll,
                        const LeadLocs &Loc) const noexcept;

        /// Makes the leader constraint matrix and RHS
        void make_LL_LeadCons(arma::sp_mat &LeadCons, arma::vec &LeadRHS,
                              const LeadAllPar &Param,
                              const Models::LeadLocs &Loc = {},
                              const unsigned int import_lim_cons = 1,
                              const unsigned int export_lim_cons = 1,
                              const unsigned int price_lim_cons = 1,
                              const unsigned int activeTaxCaps = 0) const noexcept;

        void add_Leaders_tradebalance_constraints(const unsigned int i);

        void make_MC_leader(const unsigned int i);

        void make_MC_cons(arma::sp_mat &MCLHS, arma::vec &MCRHS) const;

        void computeLeaderLocations(const bool addSpaceForMC = false);

        void add_Dummy_Lead(const unsigned int i);

        void make_obj_leader(const unsigned int i, Game::QP_objective &QP_obj);

        arma::vec sol_x, sol_z;

    public: // Attributes
        const unsigned int &nCountries{nCountr}; ///< Constant attribute for number of leaders in the EPEC
        const unsigned int &nVarEPEC{nVarinEPEC}; ///< Constant attribute for number of variables in the EPEC
        bool indicators = {
                true}; ///< Controls the flag useIndicators in LCPtoLP class. If true, indicators constraints replace bigM ones.

        EPEC() = delete;

        EPEC(GRBEnv *env, arma::sp_mat TranspCosts = {}) : TranspCosts{TranspCosts}, env{env} {}

        //Unit tests
        void testQP(const unsigned int i);

        void testLCP(const unsigned int i);

        void make_country_QP(const unsigned int i);

        void make_country_QP();

        ///@brief %Models a Standard Nash-Cournot game within a country
        EPEC &addCountry(
                /// The Parameter structure for the leader
                LeadAllPar Params,
                /// Create columns with 0s in it. To handle additional dummy leader variables.
                const unsigned int addnlLeadVars = 0
        );

        EPEC &addTranspCosts(const arma::sp_mat &costs);

        const EPEC &finalize();

        unsigned int
        getPosition(const unsigned int countryCount, const LeaderVars var = LeaderVars::FollowerStart) const;

        unsigned int getPosition(const string countryCount, const LeaderVars var = LeaderVars::FollowerStart) const;

        unique_ptr<GRBModel> Respond(const unsigned int i, const arma::vec &x) const;

        unique_ptr<GRBModel> Respond(const string name, const arma::vec &x) const;

        EPEC &unlock();

        void findNashEq();

        // Data access methods
        Game::NashGame *get_LowerLevelNash(const unsigned int i) const;

        Game::LCP *playCountry(vector<Game::LCP *> countries);

        // Writing model files
        void write(const string filename, const unsigned int i, bool append = true) const;

        void write(const string filename, bool append = true) const;

        void gur_WriteCountry_conv(const unsigned int i, string filename) const;

        void gur_WriteEpecMip(const unsigned int i, string filename) const;

        void
        WriteCountry(const unsigned int i, const string filename, const arma::vec x, const bool append = true) const;

        void WriteFollower(const unsigned int i, const unsigned int j, const string filename, const arma::vec x) const;

        void writeSolutionJSON(string filename, const arma::vec x, const arma::vec z) const ;

        void writeSolution(const int writeLevel, string filename) const;

        const arma::vec getx() const { return this->sol_x; }

        const arma::vec getz() const { return this->sol_z; }

    };


};

// Gurobi functions
string to_string(const GRBVar &var);

string to_string(const GRBConstr &cons, const GRBModel &model);

// ostream functions
namespace Models {
    enum class prn {
        label, val
    };

    ostream &operator<<(ostream &ost, Models::prn l);
};

#endif
