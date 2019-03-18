#ifndef FUNC_H
#define FUNC_H

#define VERBOSE false
#include<algorithm>
#include<iostream>
#include<ctime>
#include<vector>
#include<utility>
#include<memory>
#include<cstdlib>
#include<gurobi_c++.h>
#include<armadillo>

using namespace std;

/*****************************************************/
/* 	    										     */
/**********		FROM BALASPOLYHEDRON.CPP		******/ 
/* 	    										     */
/*****************************************************/
/// Returns a Gurobi model which can optimize over the convex hull of the 
/// union of polyhedra described in A and b where A and b are dense
GRBModel& PolyUnion(GRBModel &model, GRBVar **&x, GRBVar *&xMain, GRBVar *&delta, 
		const vector<arma::sp_mat> A, const vector<arma::vec> b);

/// Returns a Gurobi model which can optimize over the convex hull of the 
/// union of polyhedra described in A and b where A and b are sparse
GRBModel& PolyUnion(GRBModel &model, GRBVar **&x, GRBVar *&xMain, GRBVar *&delta, 
		const vector<arma::mat> A, const vector<arma::vec> b);

int PolyUnion(const vector<arma::sp_mat> Ai, const vector<arma::vec> bi, 
		arma::sp_mat& A, arma::vec &b, bool Reduce=false);

vector<unsigned int> makeCompactPolyhedron(const arma::sp_mat A, 
		const arma::vec b, arma::sp_mat &Anew, arma::vec &bnew);

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
		/// Check that the data for the QP_Param class is valid
		bool dataCheck(bool forcesymm=true) const;
		/// Initializes the size related private variables
		/// @returns Number of variables in the quadratic program, QP
		unsigned int size();
	public: // Constructors
		/// Initialize only the size. Everything else is empty (can be updated later)
		QP_Param(GRBEnv* env=nullptr):env{env},QuadModel{(*env)},made_yQy{false}{this->size();}
		/// Set data at construct time
		QP_Param(arma::sp_mat Q, arma::sp_mat C, 
				arma::sp_mat A, arma::sp_mat B, arma::vec c, arma::vec b, GRBEnv* env=nullptr):env{env},QuadModel{(*env)},made_yQy{false}
		{
			this->set(Q, C, A, B, c, b);
			this->size();
		}
		/// Copy constructor
		QP_Param(QP_Param &Qu):Q{Qu.Q}, A{Qu.A}, B{Qu.B}, C{Qu.C}, c{Qu.c}, b{Qu.b}, env{Qu.env}, QuadModel{Qu.QuadModel},made_yQy{Qu.made_yQy}{};
	public: // Set some data
		/// Setting the data, while keeping the input objects intact
		QP_Param& set(arma::sp_mat Q, arma::sp_mat C, 
				arma::sp_mat A, arma::sp_mat B, arma::vec c, arma::vec b); // Copy data into this
		/// Faster means to set data. But the input objects might be corrupted now.
		QP_Param& setMove(arma::sp_mat Q, arma::sp_mat C, 
				arma::sp_mat A, arma::sp_mat B, arma::vec c, arma::vec b); // Move data into this
	public: // Return some of the data as a copy
		/// Read-only access to the private variable Q
		inline arma::sp_mat getQ() const { return this->Q; } 
		/// Read-only access to the private variable C
		inline arma::sp_mat getC() const { return this->C; }
		/// Read-only access to the private variable A
		inline arma::sp_mat getA() const { return this->A; }
		/// Read-only access to the private variable B
		inline arma::sp_mat getB() const { return this->B; }
		/// Read-only access to the private variable c
		inline arma::vec getc() const { return this->c; }
		/// Read-only access to the private variable b
		inline arma::vec getb() const { return this->b; }
		/// Read-only access to the private variable Nx
		inline unsigned int getNx() const { return this->Nx; }
		/// Read-only access to the private variable Ny
		inline unsigned int getNy() const { return this->Ny; }
	private:
		int make_yQy();
	public: // Other methods
		/// Compute the KKT conditions for the given QP
		unsigned int KKT(arma::sp_mat& M, arma::sp_mat& N, arma::vec& q) const;
		unique_ptr<GRBModel> solveFixed(arma::vec x);
		bool is_Playable(const QP_Param P) const;
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
		/// Upper level leader constraints LHS
		arma::sp_mat LeaderConstraints;
		/// Upper level leader constraints RHS
		arma::vec LeaderConsRHS;
	public: // Variables
		/// Number of players in the Nash Game
		unsigned int Nplayers;
		/// The QP that each player solves
		vector<shared_ptr<QP_Param>> Players;
		/// Market clearing constraints
		arma::sp_mat MarketClearing;
		/// RHS to the Market Clearing constraints
		arma::vec MCRHS;			
		/// To print the Nash Game!
		friend ostream& operator<< (ostream& os, const NashGame N);
	private:
		///@internal In the vector of variables of all players,which position does the variable corrresponding to this player starts.
		vector<unsigned int> primal_position; 
		///@internal In the vector of variables of all players,which position do the DUAL variable corrresponding to this player starts.
		vector<unsigned int> dual_position; 
		///@internal Manages the position of Market clearing constraints' duals
		unsigned int MC_dual_position;
		/// @internal Manages the position of where the leader's variables start
		unsigned int Leader_position; 
		/// Number of leader variables. These many variables will not have a matching complementary equation.
		unsigned int n_LeadVar;
	public: // Constructors
		/**
		 * Construct a NashGame by giving a vector of pointers to 
		 * QP_Param, defining each player's game
		 * A set of Market clearing constraints and its RHS
		 * And if there are leader variables, the number of leader vars.
		 */
		NashGame(vector<shared_ptr<QP_Param>> Players, arma::sp_mat MC, 
				arma::vec MCRHS, unsigned int n_LeadVar=0, arma::sp_mat LeadA={}, arma::vec LeadRHS={});
		NashGame(unsigned int Nplayers, unsigned int n_LeadVar=0, arma::sp_mat LeadA={}, arma::vec LeadRHS={}): LeaderConstraints{LeadA}, LeaderConsRHS{LeadRHS}, Nplayers{Nplayers}, n_LeadVar{n_LeadVar}
		{
			Players.resize(this->Nplayers); 
			primal_position.resize(this->Nplayers);
			dual_position.resize(this->Nplayers);
		}
		/// Destructors to `delete` the QP_Param objects that might have been used.
		~NashGame();
	public: // Members
		/// Formulates the LCP corresponding to the Nash game. 
		/// @warning Does not return the leader constraints. Use NashGame::RewriteLeadCons() to handle them
		unsigned int FormulateLCP(
				///@internal Returns the \f$M\f$ corresponding to \f$Mx+q\f$
				arma::sp_mat &M, 
				///@internal Returns the \f$q\f$ corresponding to \f$Mx+q\f$
			   	arma::vec &q,
				/// There might be more variables than equations. In this case pairs the equations with variables
			   	perps &Compl, 
				/// If the solution M and q should be written to a file
				bool writeToFile = false,
				/// File names for the output M 
				string M_name = "M.txt", 
				/// File names for the output q
				string q_name = "q.txt"
				) const;
		/// Rewrites leader constraints given earlier with added empty columns and spaces corresponding to
		/// Market clearing duals and other equation duals.
		arma::sp_mat RewriteLeadCons() const;
};

// void MPEC(NashGame N, arma::sp_mat Q, QP_Param &P);
ostream& operator<< (ostream& os, const QP_Param &Q);


/************************************************/
/* 	 									      	*/
/*******			FROM LCPTOLP.CPP	    *****/ 
/* 	 									      	*/
/************************************************/

/// Checks if the polyhedron given by @f$ Ax\leq b@f$ is feasible.
/// If yes, returns the point @f$x@f$ in the polyhedron that minimizes @f$c^Tx@f$
arma::vec isFeas(const arma::sp_mat* A, const arma::vec *b, 
		const arma::vec *c, bool Positivity=false);

/** 
 * Computes the convex hull of a finite union of polyhedra where 
 * each polyhedra @f$P_i@f$ is of the form
 * @f{eqnarray}{
 * A^ix &\leq& b^i\\
 * x &\geq& 0
 * @f}
*/
int ConvexHull(
		/// Inequality constraints LHS that define polyhedra whose convex hull is to be found
		vector<arma::sp_mat*> *Ai, 
		/// Inequality constraints RHS that define polyhedra whose convex hull is to be found
		vector<arma::vec*> *bi, 
		/// Pointer to store the output of the convex hull LHS
		arma::sp_mat &A, 
		/// Pointer to store the output of the convex hull RHS
		arma::vec &b, 
		/// Any common constraints to ALL the polyhedra - LHS.
		arma::sp_mat Acom={},
		/// Any common constraints to ALL the polyhedra - RHS.
	   	arma::vec bcom={} 
		);

/**
 * @brief Class to handle and solve linear complementarity problems
 */
/**
* A class to handle linear complementarity problems (LCP)
* especially as MIPs with bigM constraints
* Also provides the convex hull of the feasible space!
*/
class LCP
{
	private:
	// Essential data
		/// Gurobi environment for MIP/LP solves
		GRBEnv* env;
		/// M in @f$Mx+q@f$ that defines the LCP
		arma::sp_mat M; 
		/// q in @f$Mx+q@f$ that defines the LCP
		arma::vec q; 
		/// Compl stores data in <Eqn, Var> form.
		perps Compl; 
		unsigned int LeadStart, LeadEnd, nLeader; 
		/// Apart from @f$0 \le x \perp Mx+q\ge 0@f$, one needs@f$ Ax\le b@f$ too!
		arma::sp_mat _A; arma::vec _b;		
	// Temporary data
		/// Keep track if LCP::RlxdModel is made
		bool madeRlxdModel;
		unsigned int nR, nC;
		/// LCP feasible region is a union of polyhedra. Keeps track which of those inequalities are fixed to equality to get the individual polyhedra
		vector<vector<short int>*>* AllPolyhedra;
		vector<arma::sp_mat*>* Ai; vector<arma::vec*>* bi;
		// A gurobi model with all complementarity constraints removed.
	   	GRBModel RlxdModel;
	public: 
	// Fudgible data
		/// bigM used to rewrite the LCP as MIP
		long double bigM;
		/// The threshold, below which a number would be considered to be zero.
		long double eps;
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
		/// Read-only access to LCP::M
		inline arma::sp_mat getM() {return this->M;}  
		/// Pointer access to LCP::M
		inline arma::sp_mat* getMstar() {return &(this->M);}
		/// Read-only access to LCP::q
		inline arma::vec getq() {return this->q;}  
		/// Pointer access to LCP::q
		inline arma::vec* getqstar() {return &(this->q);}
		/// Read-only access to LCP::LeadStart
		inline unsigned int getLStart(){return LeadStart;} 
		/// Read-only access to LCP::LeadEnd
		inline unsigned int getLEnd(){return LeadEnd;}
		/// Read-only access to LCP::Compl
		inline perps getCompl() {return this->Compl;}  
		/// Print a summary of the LCP
		void print(string end="\n");
	/* Member functions */
	private:
		bool errorCheck(bool throwErr=true) const;
		void defConst(GRBEnv* env);
		int makeRelaxed();
	public:
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
		int BranchLoc(unique_ptr<GRBModel> &m, vector<short int>* Fix);
		int BranchProcLoc(vector<short int>* Fix, vector<short int> *Leaf);
		int EnumerateAll(bool solveLP=false);
	public:
		bool extractSols(GRBModel* model, arma::vec &z, 
				arma::vec &x, bool extractZ = false) const; 
		vector<vector<short int>*> *BranchAndPrune ();
		unique_ptr<GRBModel> LCPasQP(bool solve = false);
	/* Convex hull computation */
	private:
		void FixToPoly(const vector<short int> *Fix, bool checkFeas = false);
		void FixToPolies(const vector<short int> *Fix, bool checkFeas = false);
	public:
		/**
		 * Computes the convex hull of the feasible region of the LCP
		 * @warning To be run only after LCP::BranchAndPrune is run. Otherwise this can give errors
		 * @todo Formally call LCP::BranchAndPrune or throw an exception if this method is not already run
		 */
		int ConvexHull(
				/// Convex hull inequality description LHS to be stored here
				arma::sp_mat& A,
				/// Convex hull inequality description RHS to be stored here
			   	arma::vec &b) 
		{return ::ConvexHull(this->Ai, this->bi, A, b, this->_A,this->_b);};
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
	/// Quadratic coefficient of i-th follower's cost. Size of this vector should be equal to n_followers
	vector<double> costs_quad;
	/// Linear  coefficient of i-th follower's cost. Size of this vector should be equal to n_followers
	vector<double> costs_lin;
	/// Production capacity of each follower. Size of this vector should be equal to n_followers
	vector<double> capacities;
};


/// @brief Stores the parameters of the demand curve in a country model
struct DemPar
{
	/// Intercept of the demand curve. Written as: Price = alpha - beta*(Total quantity in domestic market) 
	double alpha = 100;
	/// Slope of the demand curve. Written as: Price = alpha - beta*(Total quantity in domestic market) 
	double beta = 2;
	DemPar(double alpha=100, double beta=2):alpha{alpha}, beta{beta}{};
};

/// @brief Stores the parameters of the leader in a country model
struct LeadPar
{
	/// Maximum net import in the country. If no limit, set the value as -1;
	double import_limit = -1; 
	/// Maximum net export in the country. If no limit, set the value as -1;
	double export_limit = -1;
	/// Government decided increase in the shift in costs_lin of any player cannot exceed this value
	double max_tax_perc = 0.3;
	LeadPar(double max_tax_perc=0.3, double imp_lim=-1, double exp_lim=-1):import_limit{imp_lim}, export_limit{exp_lim}, max_tax_perc{max_tax_perc}{}
};

/// @brief Stores the parameters of a country model
struct LeadAllPar
{
	/// Number of followers in the country
	unsigned int n_followers;
	/// A struct to hold Follower Parameters
	Models::FollPar FollowerParam = {};
	/// A struct to hold Demand Parameters
	Models::DemPar DemandParam = {};
	/// A struct to hold Leader Parameters
	Models::LeadPar LeaderParam = {};
	LeadAllPar(unsigned int n, Models::FollPar FP={}, Models::DemPar DP={}, Models::LeadPar LP={}):n_followers{n}, FollowerParam{FP}, DemandParam{DP}, LeaderParam{LP}{};
};


ostream& operator<<(ostream& ost, const FollPar P);
ostream& operator<<(ostream& ost, const DemPar P);
ostream& operator<<(ostream& ost, const LeadPar P);
ostream& operator<<(ostream& ost, const LeadAllPar P);


///@brief %Models a Standard Nash-Cournot game within a country
LCP* createCountry(
		/// A gurobi environment to create and process the resulting LCP object.
		GRBEnv env, 
		/// The Parameter structure for the leader
		LeadAllPar Params, 
		/// Create columns with 0s in it. To handle additional dummy leader variables.
		const unsigned int addnlLeadVars  = 0
		);

LCP* playCountry(
		vector<LCP*> countries,
		vector<Models::LeadAllPar> Pi
		);

};
// End of namespace Models {
#endif
