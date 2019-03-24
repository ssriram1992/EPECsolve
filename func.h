#ifndef FUNC_H
#define FUNC_H

#define VERBOSE false
#include<algorithm>
#include<map>
#include<iostream>
#include<ctime>
#include<vector>
#include<utility>
#include<memory>
#include<cstdlib>
#include<gurobi_c++.h>
#include<armadillo>

using namespace std;


/********************************************/
/* 	    							    	*/
/******			FROM GAMES.CPP		  *******/ 
/* 	    							    	*/
/********************************************/

using perps = vector<pair<unsigned int, unsigned int>>  ;
ostream& operator<<(ostream& ost, perps C);
inline bool operator < (vector<int> Fix1, vector<int> Fix2);
inline bool operator == (vector<int> Fix1, vector<int> Fix2);
template <class T> ostream& operator<<(ostream& ost, vector<T> v);
template <class T, class S> ostream& operator<<(ostream& ost, pair<T,S> p);

namespace Game{
///@brief Class to handle parameterized quadratic programs(QP)
class QP_Param
/**
 * Represents a Parameterized QP as \f[
 * \min_y \frac{1}{2}y^TQy + c^Ty + (Cx)^T y
 * \f]
 * Subject to
 * \f{eqnarray}{
 * Ax + By &\leq& b \\
 * y &\geq& 0
 * \f}
*/
{
	private: // Data representing the parameterized QP
		arma::sp_mat Q, A, B, C;
		arma::vec c, b;
	private: // Other private objects
		GRBEnv *env;
		GRBModel QuadModel;
		bool made_yQy;
		unsigned int Nx, Ny, Ncons;
		bool dataCheck(bool forcesymm=true) const;
		unsigned int size();
	public: // Constructors
		/// Initialize only the size. Everything else is empty (can be updated later)
		QP_Param(GRBEnv* env=nullptr):env{env},QuadModel{(*env)},made_yQy{false}{this->size();}
		/// Set data at construct time
		QP_Param(arma::sp_mat Q, arma::sp_mat C, arma::sp_mat A, arma::sp_mat B, arma::vec c, arma::vec b, 
				GRBEnv* env=nullptr):env{env},QuadModel{(*env)},made_yQy{false} 
		{
			this->set(Q, C, A, B, c, b);
			this->size();
			if(!this->dataCheck()) throw string("Error in QP_Param::QP_Param: Invalid data for constructor");
		}
		/// Copy constructor
		QP_Param(QP_Param &Qu):Q{Qu.Q}, A{Qu.A}, B{Qu.B}, C{Qu.C}, c{Qu.c}, b{Qu.b}, 
				env{Qu.env}, QuadModel{Qu.QuadModel},made_yQy{Qu.made_yQy}{this->size();};
	public: // Set some data
		QP_Param& set(arma::sp_mat &Q, arma::sp_mat &C, 
				arma::sp_mat &A, arma::sp_mat &B, arma::vec &c, arma::vec &b); // Copy data into this
		QP_Param& set(arma::sp_mat &&Q, arma::sp_mat &&C, 
				arma::sp_mat &&A, arma::sp_mat &&B, arma::vec &&c, arma::vec &&b); // Move data into this
	public: // Return some of the data as a copy 
		inline arma::sp_mat getQ() const { return this->Q; } 		///< Read-only access to the private variable Q 
		inline arma::sp_mat getC() const { return this->C; }		///< Read-only access to the private variable C 
		inline arma::sp_mat getA() const { return this->A; }		///< Read-only access to the private variable A 
		inline arma::sp_mat getB() const { return this->B; }		///< Read-only access to the private variable B 
		inline arma::vec getc() const { return this->c; }			///< Read-only access to the private variable c 
		inline arma::vec getb() const { return this->b; }			///< Read-only access to the private variable b 
		inline unsigned int getNx() const { return this->Nx; }		///< Read-only access to the private variable Nx 
		inline unsigned int getNy() const { return this->Ny; }		///< Read-only access to the private variable Ny
	private:
		int make_yQy();
	public: // Other methods
		unsigned int KKT(arma::sp_mat& M, arma::sp_mat& N, arma::vec& q) const;
		unique_ptr<GRBModel> solveFixed(arma::vec x);
		inline bool is_Playable(const QP_Param &P) const 
		/// Checks if the current object can play a game with another Game::QP_Param object @p P.
		{
			bool b1, b2, b3;
			b1 = (this->Nx + this-> Ny ) == (P.getNx()+P.getNy());
			b2 = this->Nx >= P.getNy();
			b3 = this->Ny <= P.getNx();
			return b1&&b2&&b3;
		}
		QP_Param& addDummy(unsigned int pars, unsigned int vars = 0);
};

/**
 * @brief Class to model Nash-cournot games with each player playing a QP
 */
/**
 * Stores a vector of QPs with each player's optimization problem.
 * Potentially common (leader) constraints can be stored too.
 *
 * Helpful in rewriting the Nash-Cournot game as an LCP
 * Helpful in rewriting leader constraints after incorporating dual variables etc
 * @warning This has public fields which if accessed and changed can cause
 * undefined behavior! 
 * \todo Better implementation which will make the above warning go away!
 */
class NashGame
{
	private: 
		arma::sp_mat LeaderConstraints;		///< Upper level leader constraints LHS 
		arma::vec LeaderConsRHS;			///< Upper level leader constraints RHS 
		unsigned int Nplayers;				///< Number of players in the Nash Game 
		vector<shared_ptr<QP_Param>> Players;	///< The QP that each player solves 
		arma::sp_mat MarketClearing;			///< Market clearing constraints 
		arma::vec MCRHS;						///< RHS to the Market Clearing constraints

	private:
		/// @internal In the vector of variables of all players,
		/// which position does the variable corrresponding to this player starts.
		vector<unsigned int> primal_position; 
		///@internal In the vector of variables of all players,
		/// which position do the DUAL variable corrresponding to this player starts.
		vector<unsigned int> dual_position; 
		/// @internal Manages the position of Market clearing constraints' duals
		unsigned int MC_dual_position;
		/// @internal Manages the position of where the leader's variables start
		unsigned int Leader_position; 
		/// Number of leader variables. 
		/// These many variables will not have a matching complementary equation.
		unsigned int n_LeadVar;

	public: // Constructors
		NashGame(vector<shared_ptr<QP_Param>> Players, arma::sp_mat MC, 
				arma::vec MCRHS, unsigned int n_LeadVar=0, arma::sp_mat LeadA={}, arma::vec LeadRHS={});
		NashGame(unsigned int Nplayers, unsigned int n_LeadVar=0, arma::sp_mat LeadA={}, arma::vec LeadRHS={})
			:LeaderConstraints{LeadA}, LeaderConsRHS{LeadRHS}, Nplayers{Nplayers}, n_LeadVar{n_LeadVar}
		{
			Players.resize(this->Nplayers); 
			primal_position.resize(this->Nplayers);
			dual_position.resize(this->Nplayers);
		}
		/// Destructors to `delete` the QP_Param objects that might have been used.
		~NashGame(){};
	
	private:
		void set_positions();

	public: 
		friend ostream& operator<< (ostream& os, const NashGame &N)		///< To print the Nash Game!
		{
			os<<endl;
			os<<"-----------------------------------------------------------------------"<<endl;
			os<<"Nash Game with "<<N.Nplayers<<" players"<<endl;
			os<<"-----------------------------------------------------------------------"<<endl;
			os<<"Number of primal variables:\t\t\t "<<N.primal_position.back()<<endl;
			os<<"Number of dual variables:\t\t\t "<<N.dual_position.back()-N.dual_position.front()+1<<endl;
			os<<"Number of shadow price dual variables:\t\t "<<N.MCRHS.n_rows<<endl;
			os<<"Number of leader variables:\t\t\t "<<N.n_LeadVar<<endl;
			os<<"-----------------------------------------------------------------------"<<endl;
			return os;
		}
		/// Return the number of primal variables
		inline unsigned int getNprimals() const { return this->Players.at(0)->getNy() + this->Players.at(0)->getNx(); }


	public: // Members
		const NashGame& FormulateLCP(arma::sp_mat &M, arma::vec &q,	perps &Compl, bool writeToFile = false,	string M_name = "M.txt", string q_name = "q.txt") const; 
		arma::sp_mat RewriteLeadCons() const;
		inline arma::vec getLeadRHS() const {return this->LeaderConsRHS;}
		NashGame& addDummy(unsigned int par=0);
		NashGame& addLeadCons(const arma::vec &a, double b);
};



// void MPEC(NashGame N, arma::sp_mat Q, QP_Param &P);
ostream& operator<< (ostream& os, const QP_Param &Q);
};

/************************************************/
/* 	 									      	*/
/*******			FROM LCPTOLP.CPP	    *****/ 
/* 	 									      	*/
/************************************************/



namespace Game{

arma::vec LPSolve(const arma::sp_mat &A, const arma::vec &b, const arma::vec &c, int &status, bool Positivity=false);
int ConvexHull( vector<arma::sp_mat*> *Ai, vector<arma::vec*> *bi, arma::sp_mat &A, arma::vec &b, arma::sp_mat Acom={}, arma::vec bcom={});
/**
 * @brief Class to handle and solve linear complementarity problems
 */
/**
* A class to handle linear complementarity problems (LCP)
* especially as MIPs with bigM constraints
* Also provides the convex hull of the feasible space, restricted feasible space etc.
*/
class LCP
{
	private:
	// Essential data ironment for MIP/LP solves
		GRBEnv* env;			///< Gurobi env 
		arma::sp_mat M; 		///< M in @f$Mx+q@f$ that defines the LCP 
		arma::vec q; 			///< q in @f$Mx+q@f$ that defines the LCP 
		perps Compl; 			///< Compl stores data in <Eqn, Var> form.
		unsigned int LeadStart, LeadEnd, nLeader; 
		arma::sp_mat _A; arma::vec _b;	///< Apart from @f$0 \le x \perp Mx+q\ge 0@f$, one needs@f$ Ax\le b@f$ too!
	// Temporary data 
		bool madeRlxdModel;	///< Keep track if LCP::RlxdModel is made
		unsigned int nR, nC;
		/// LCP feasible region is a union of polyhedra. Keeps track which of those inequalities are fixed to equality to get the individual polyhedra
		vector<vector<short int>*> *AllPolyhedra, *RelAllPol;
		vector<arma::sp_mat*> *Ai, *Rel_Ai; vector<arma::vec*> *bi, *Rel_bi; 
	   	GRBModel RlxdModel;	///< A gurobi model with all complementarity constraints removed.
	public: 
	// Fudgible data 
		long double bigM;	///< bigM used to rewrite the LCP as MIP 
		long double eps;	///< The threshold, below which a number would be considered to be zero.
	public:
	/** Constructors */
		/// Class has no default constructors
		LCP() = delete;	
		LCP(GRBEnv* env, arma::sp_mat M, arma::vec q, 
				unsigned int LeadStart, unsigned LeadEnd, arma::sp_mat A={}, arma::vec b={}); // Constructor with M,q,leader posn
		LCP(GRBEnv* env, arma::sp_mat M, arma::vec q, 
				perps Compl, arma::sp_mat A={}, arma::vec b={}); // Constructor with M, q, compl pairs
		LCP(GRBEnv* env, NashGame N);
	/** Destructor - to delete the objects created with new operator */
		~LCP();
	/** Return data and address */ 
		inline arma::sp_mat getM() {return this->M;}  			///< Read-only access to LCP::M 
		inline arma::sp_mat* getMstar() {return &(this->M);}	///< Pointer access to LCP::M 
		inline arma::vec getq() {return this->q;}  				///< Read-only access to LCP::q 
		inline arma::vec* getqstar() {return &(this->q);}		///< Pointer access to LCP::q 
		inline unsigned int getLStart(){return LeadStart;} 		///< Read-only access to LCP::LeadStart 
		inline unsigned int getLEnd(){return LeadEnd;}			///< Read-only access to LCP::LeadEnd 
		inline perps getCompl() {return this->Compl;}  			///< Read-only access to LCP::Compl 
		void print(string end="\n");							///< Print a summary of the LCP
	/* Member functions */
	private:
		bool errorCheck(bool throwErr=true) const;
		void defConst(GRBEnv* env);
		void makeRelaxed();
	/* Solving relaxations and restrictions */
	private:
		unique_ptr<GRBModel> LCPasMIP(vector<unsigned int> FixEq={}, 
				vector<unsigned int> FixVar={}, bool solve=false);
		unique_ptr<GRBModel> LCPasMIP(vector<short int> Fixes, bool solve);
		unique_ptr<GRBModel> LCP_Polyhed_fixed(vector<unsigned int> FixEq={}, 
				vector<unsigned int> FixVar={});
		unique_ptr<GRBModel> LCP_Polyhed_fixed(arma::Col<int> FixEq, 
				arma::Col<int> FixVar);
	/* Branch and Prune Methods */
	private:
		template<class T> inline bool isZero(const T val) const { return (val>-eps && val < eps);}
		inline vector<short int>* solEncode(GRBModel *model)const;
		vector<short int>* solEncode(const arma::vec &z, const arma::vec &x)const;
		void branch(int loc, const vector<short int> *Fixes);
		vector<short int>* anyBranch(const vector<vector<short int>*>* vecOfFixes, vector<short int>* Fix) const;
		int branchLoc(unique_ptr<GRBModel> &m, vector<short int>* Fix);
		int branchProcLoc(vector<short int>* Fix, vector<short int> *Leaf);
		LCP& EnumerateAll(bool solveLP=false);
	public:
		bool extractSols(GRBModel* model, arma::vec &z, 
				arma::vec &x, bool extractZ = false) const; 
		vector<vector<short int>*> *BranchAndPrune ();
	/* Getting single point solutions */
	public:
		unique_ptr<GRBModel> LCPasQP(bool solve = false);
		unique_ptr<GRBModel> LCPasMIP(bool solve = false);
		unique_ptr<GRBModel> MPECasMILP(const arma::sp_mat &C, const arma::vec &c, const arma::vec &x_minus_i, bool solve = false);
		unique_ptr<GRBModel> MPECasMIQP(const arma::sp_mat &Q, const arma::sp_mat &C, const arma::vec &c, const arma::vec &x_minus_i, bool solve = false);
	/* Convex hull computation */
	private:
		LCP& FixToPoly(const vector<short int> *Fix, bool checkFeas = false, bool custom=false, vector<arma::sp_mat*> *custAi={}, vector<arma::vec*> *custbi={});
		LCP& FixToPolies(const vector<short int> *Fix, bool checkFeas = false, bool custom=false, vector<arma::sp_mat*> *custAi={}, vector<arma::vec*> *custbi={});
	public:
		LCP& addPolyhedron(const vector<short int> &Fix, vector<arma::sp_mat*> &custAi, vector<arma::vec*> &custbi, 
				const bool convHull = false, arma::sp_mat *A={}, arma::vec  *b={});
		int ConvexHull( 
				arma::sp_mat& A,		///< Convex hull inequality description LHS to be stored here 
			   	arma::vec &b) 			///< Convex hull inequality description RHS to be stored here
		/**
		 * Computes the convex hull of the feasible region of the LCP
		 * @warning To be run only after LCP::BranchAndPrune is run. Otherwise this can give errors
		 * @todo Formally call LCP::BranchAndPrune or throw an exception if this method is not already run
		 */
		{return Game::ConvexHull(this->Ai, this->bi, A, b, this->_A,this->_b);};
};
};


/************************************************/
/* 	 									      	*/
/*******			FROM Models.CPP	        *****/ 
/* 	 									      	*/
/************************************************/
namespace Models{
typedef struct FollPar FollPar; 
typedef struct DemPar DemPar;
typedef struct LeadPar LeadPar;
typedef struct LeadAllPar LeadAllPar;

/// @brief Stores the parameters of the follower in a country model
struct FollPar
{ 
	vector<double> costs_quad = {};	///< Quadratic coefficient of i-th follower's cost. Size of this vector should be equal to n_followers 
	vector<double> costs_lin = {};	///< Linear  coefficient of i-th follower's cost. Size of this vector should be equal to n_followers 
	vector<double> capacities = {};	///< Production capacity of each follower. Size of this vector should be equal to n_followers 
	vector<double> emission_costs = {};	///< Emission costs for unit quantity of the fuel. Emission costs feature only on the leader's problem 
	vector<string> names = {};	///< Optional Names for the Followers.
};


/// @brief Stores the parameters of the demand curve in a country model
struct DemPar
{ 
	double alpha = 100;	///< Intercept of the demand curve. Written as: Price = alpha - beta*(Total quantity in domestic market) 
	double beta = 2;	///< Slope of the demand curve. Written as: Price = alpha - beta*(Total quantity in domestic market) 
	DemPar(double alpha=100, double beta=2):alpha{alpha}, beta{beta}{};
};

/// @brief Stores the parameters of the leader in a country model
struct LeadPar
{ 
	double import_limit = -1; 	///< Maximum net import in the country. If no limit, set the value as -1; 
	double export_limit = -1;	///< Maximum net export in the country. If no limit, set the value as -1; 
	double max_tax_perc = 0.3;	///< Government decided increase in the shift in costs_lin of any player cannot exceed this value 
	double price_limit = -1;	///< Government does not want the price to exceed this limit
	LeadPar(double max_tax_perc=0.3, double imp_lim=-1, double exp_lim=-1):import_limit{imp_lim}, export_limit{exp_lim}, max_tax_perc{max_tax_perc}{}
};

/// @brief Stores the parameters of a country model
struct LeadAllPar
{ 
	unsigned int n_followers;			///< Number of followers in the country 
	string name;						///< Country Name 
	Models::FollPar FollowerParam = {};	///< A struct to hold Follower Parameters 
	Models::DemPar DemandParam = {};	///< A struct to hold Demand Parameters 
	Models::LeadPar LeaderParam = {};	///< A struct to hold Leader Parameters
	LeadAllPar(unsigned int n_foll, string name, Models::FollPar FP={}, Models::DemPar DP={}, Models::LeadPar LP={}):n_followers{n_foll}, name{name}, FollowerParam{FP}, DemandParam{DP}, LeaderParam{LP}
	{
		// Nothing here
	}
};


ostream& operator<<(ostream& ost, const FollPar P);
ostream& operator<<(ostream& ost, const DemPar P);
ostream& operator<<(ostream& ost, const LeadPar P);
ostream& operator<<(ostream& ost, const LeadAllPar P);




class EPEC
{
	private:
		vector<LeadAllPar> AllLeadPars = {};  ///< The parameters of each leader in the EPEC game
		vector<shared_ptr<Game::NashGame>> countriesLL = {}; ///< Stores each country's lower level Nash game
		vector<arma::sp_mat> LeadConses = {}; ///< Stores each country's leader constraint LHS
		vector<arma::vec> LeadRHSes = {}; ///< Stores each country's leader constraint RHS
		arma::sp_mat TranspCosts = {};
		vector<unsigned int> nImportMarkets = {};
	private:
		GRBEnv *env;		///< A gurobi environment to create and process the resulting LCP object.
		map<string, unsigned int> name2nos = {};
		bool finalized = false;
	public:
		EPEC()=delete;
		EPEC(GRBEnv *env, arma::sp_mat TranspCosts={}):TranspCosts{TranspCosts}, env{env}{}
	private:
		/// Checks that the parameter given to add a country is valid. Does not have obvious errors
		bool ParamValid(const LeadAllPar& Param) const;
		/// Makes the lower level quadratic program object for each follower.
		void make_LL_QP(const LeadAllPar& Params, 
				const unsigned int follower, 
				Game::QP_Param* Foll, 
				const unsigned int LeadVars) const noexcept;
		/// Makes the leader constraint matrix and RHS
		void make_LL_LeadCons(arma::sp_mat &LeadCons, arma::vec &LeadRHS,
				const LeadAllPar& Param,
				const unsigned int import_lim_cons=1,
				const unsigned int export_lim_cons=1,
				const unsigned int price_lim_cons=1
				) const noexcept;
	public:
		///@brief %Models a Standard Nash-Cournot game within a country
		EPEC& addCountry(
				/// The Parameter structure for the leader
				LeadAllPar Params, 
				/// Create columns with 0s in it. To handle additional dummy leader variables.
				const unsigned int addnlLeadVars  = 0
				);
		EPEC& addTranspCosts(const arma::sp_mat& costs);
		EPEC& finalize();
	public:
		// Data access methods
		Game::NashGame* get_LowerLevelNash(unsigned int i); 
		Game::LCP* playCountry(vector<Game::LCP*> countries);
};




};
// End of namespace Models {
#endif
