#ifndef FUNC_H
#define FUNC_H

#define VERBOSE false
#include<algorithm>
#include<iostream>
#include<ctime>
#include<vector>
#include<utility>
#include<cstdlib>
#include<gurobi_c++.h>
#include<armadillo>

using namespace std;

/*****************************************************/
/* 	    										     */
/**********		FROM BALASPOLYHEDRON.CPP		******/ 
/* 	    										     */
/*****************************************************/
// Returns a Gurobi model which can optimize over the convex hull of the 
// union of polyhedra described in A and b where A and b are dense
GRBModel& PolyUnion(GRBModel &model, GRBVar **&x, GRBVar *&xMain, GRBVar *&delta, 
		const vector<arma::sp_mat> A, const vector<arma::vec> b);

// Returns a Gurobi model which can optimize over the convex hull of the 
// union of polyhedra described in A and b where A and b are sparse
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


class QP_Param
/**
 * Represents a Parameterized QP as
 * \min_y \frac{1}{2}y^TQy + c^Ty + (Cx)^T y
 * Subject to
 * Ax + By <= b
 * y >= 0
*/
{
	private: // Data representing the parameterized QP
		arma::sp_mat Q, C, A, B;
		arma::vec c, b;
	private: // Other private objects
		unsigned int Nx, Ny, Ncons;
		bool dataCheck(bool forcesymm=true) const;
		unsigned int size();
	public: // Constructors
		QP_Param(){this->size();}
		QP_Param(arma::sp_mat Q, arma::sp_mat C, 
				arma::sp_mat A, arma::sp_mat B, arma::vec c, arma::vec b)
		{
			this->set(Q, C, A, B, c, b);
			this->size();
		}
	public: // Set some data
		QP_Param& set(arma::sp_mat Q, arma::sp_mat C, 
				arma::sp_mat A, arma::sp_mat B, arma::vec c, arma::vec b); // Copy data into this
		QP_Param& setMove(arma::sp_mat Q, arma::sp_mat C, 
				arma::sp_mat A, arma::sp_mat B, arma::vec c, arma::vec b); // Move data into this
	public: // Return some of the data as a copy
		inline arma::sp_mat getQ() const { return this->Q; } 
		inline arma::sp_mat getC() const { return this->C; }
		inline arma::sp_mat getA() const { return this->A; }
		inline arma::sp_mat getB() const { return this->B; }
		inline arma::vec getc() const { return this->c; }
		inline arma::vec getb() const { return this->b; }
		inline unsigned int getNx() const { return this->Nx; }
		inline unsigned int getNy() const { return this->Ny; }
	public: // Other methods
		unsigned int KKT(arma::sp_mat& M, arma::sp_mat& N, arma::vec& q) const;
		bool is_Playable(const QP_Param P) const;
};

class NashGame
/**
 * NashGame(vector<QP_Param*> Players, arma::sp_mat MC, 
				arma::vec MCRHS, unsigned int n_LeadVar=0);
 * Construct a NashGame by giving a vector of pointers to 
 * QP_Param, defining each player's game
 * A set of Market clearing constraints and its RHS
 * And if there are leader variables, the number of leader vars.
 */
{
	public: // Variables
		arma::sp_mat LeaderConstraints;
		arma::vec LeaderConsRHS;
		unsigned int Nplayers;
		vector<QP_Param*> Players;
		arma::sp_mat MarketClearing;
		arma::vec MCRHS;			// RHS to the Market Clearing constraints
		// In the vector of variables of all players,
		// which position does the variable corrresponding to this player starts.
		vector<unsigned int> primal_position; 
		vector<unsigned int> dual_position; 
		unsigned int MC_dual_position;
		unsigned int Leader_position; // Position from where leader's vars start
		unsigned int n_LeadVar;
	public: // Constructors
		NashGame(vector<QP_Param*> Players, arma::sp_mat MC, 
				arma::vec MCRHS, unsigned int n_LeadVar=0, arma::sp_mat LeadA={}, arma::vec LeadRHS={});
		NashGame(unsigned int Nplayers, unsigned int n_LeadVar=0, arma::sp_mat LeadA={}, arma::vec LeadRHS={}): LeaderConstraints{LeadA}, LeaderConsRHS{LeadRHS}, Nplayers{Nplayers}, n_LeadVar{n_LeadVar}
		{
			Players.resize(this->Nplayers); 
			primal_position.resize(this->Nplayers);
			dual_position.resize(this->Nplayers);
		}
		~NashGame();
	public: // Members
		unsigned int FormulateLCP(arma::sp_mat &M, arma::vec &q, perps &Compl) const;
		arma::sp_mat RewriteLeadCons() const;
};

// void MPEC(NashGame N, arma::sp_mat Q, QP_Param &P);
ostream& operator<< (ostream& os, const QP_Param &Q);
ostream& operator<< (ostream& os, const NashGame N);


/************************************************/
/* 	 									      	*/
/*******			FROM LCPTOLP.CPP	    *****/ 
/* 	 									      	*/
/************************************************/

arma::vec* isFeas(const arma::sp_mat* A, const arma::vec *b, 
		const arma::vec *c, bool Positivity=false);

int ConvexHull(
		vector<arma::sp_mat*> *Ai, vector<arma::vec*> *bi, // Individual constraints
		arma::sp_mat *A, arma::vec *b, // To store outputs
		arma::sp_mat Acom={}, arma::vec bcom={} // Common constraints.
		);

class LCP
{
	/**
	* A class to handle linear complementarity problems (LCP)
	* especially as MIPs with bigM constraints
	*/
	private:
	// Essential data
		GRBEnv* env;
		arma::sp_mat M; arma::vec q; perps Compl; /// Compl stores data in <Eqn, Var> form.
		unsigned int LeadStart, LeadEnd, nLeader; /// Positions and sizes of Leader variables
		arma::sp_mat _A; arma::vec _b;		/// Apart from 0 \leq x \perp Mx+q\geq 0, one needs Ax\leq b too!
	// Temporary data
		bool madeRlxdModel;
		unsigned int nR, nC;
		vector<vector<short int>*>* AllPolyhedra;
		vector<arma::sp_mat*>* Ai; vector<arma::vec*>* bi;
	   	GRBModel RlxdModel;
	public: 
	// Fudgible data
		long double bigM;
		long double eps;
	public:
	/** Constructors */
		LCP() = delete;	/// Class has no default constructors
		LCP(GRBEnv* env, arma::sp_mat M, arma::vec q, 
				unsigned int LeadStart, unsigned LeadEnd, arma::sp_mat A={}, arma::vec b={}); // Constructor with M,q,leader posn
		LCP(GRBEnv* env, arma::sp_mat M, arma::vec q, 
				perps Compl, arma::sp_mat A={}, arma::vec b={}); // Constructor with M, q, compl pairs
		LCP(GRBEnv* env, NashGame N);
	/** Destructor - to delete the objects created with new operator */
		~LCP();
	/** Return data and address */
		inline arma::sp_mat getM() {return this->M;}  
		inline arma::sp_mat* getMstar() {return &(this->M);}
		inline arma::vec getq() {return this->q;}  
		inline arma::vec* getqstar() {return &(this->q);}
		inline unsigned int getLStart(){return LeadStart;} 
		inline unsigned int getLEnd(){return LeadEnd;}
		inline perps getCompl() {return this->Compl;}  
		void print(string end="\n");
	/* Member functions */
	private:
		bool errorCheck(bool throwErr=true) const;
		void defConst(GRBEnv* env);
		int makeRelaxed();
	public:
		GRBModel* LCPasMIP(vector<unsigned int> FixEq={}, 
				vector<unsigned int> FixVar={}, bool solve=false);
		GRBModel* LCPasMIP(vector<short int> Fixes, bool solve);
		GRBModel* LCP_Polyhed_fixed(vector<unsigned int> FixEq={}, 
				vector<unsigned int> FixVar={});
		GRBModel* LCP_Polyhed_fixed(arma::Col<int> FixEq, 
				arma::Col<int> FixVar);
	/* Branch and Prune Methods */
	private:
		template<class T> inline bool isZero(const T val) const { return (val>-eps && val < eps);}
		inline vector<short int>* solEncode(GRBModel *model)const;
		vector<short int>* solEncode(const arma::vec &z, const arma::vec &x)const;
		void branch(int loc, const vector<short int> *Fixes);
		vector<short int>* anyBranch(const vector<vector<short int>*>* vecOfFixes, vector<short int>* Fix) const;
		int BranchLoc(GRBModel* m, vector<short int>* Fix);
		int BranchProcLoc(vector<short int>* Fix, vector<short int> *Leaf);
		int EnumerateAll(bool solveLP=false);
	public:
		bool extractSols(GRBModel* model, arma::vec &z, 
				arma::vec &x, bool extractZ = false) const; 
		vector<vector<short int>*> *BranchAndPrune ();
	/* Convex hull computation */
	private:
		void FixToPoly(const vector<short int> *Fix, bool checkFeas = false);
		void FixToPolies(const vector<short int> *Fix, bool checkFeas = false);
	public:
		int ConvexHull(arma::sp_mat* A, arma::vec *b) 
		{return ::ConvexHull(this->Ai, this->bi, A, b, this->_A,this->_b);};
};



int LCPasLPTree(
		const arma::sp_mat M,
	   	const arma::sp_mat N,
		const arma::vec q,
		vector<arma::sp_mat> &A,
		vector<arma::vec> &b,
	   	vector<arma::vec> &sol,
		bool cleanup);
int LCPasLP(
		const arma::sp_mat M,
		const arma::vec q,
		vector<arma::sp_mat> &A,
		vector<arma::vec> &b,
	   	vector<arma::vec> &sol,
		bool cleanup ,
		bool Gurobiclean );
int BinaryArr(int *selecOfTwo, unsigned int size, long long unsigned int i);
bool isEmpty(const arma::sp_mat A, const arma::vec b, arma::vec &sol);

/************************************************/
/* 	 									      	*/
/*******			FROM Models.CPP	        *****/ 
/* 	 									      	*/
/************************************************/
namespace Models{
/******************************************************************************************************
 *  
 * INPUTS:
 * 		n_followers			: (int) Number of followers in the country
 * 		costs_quad			: (vector<double>) Quadratic coefficient of i-th follower's
 * 							  cost. Size of this vector should be equal to n_followers
 * 		costs_lin			: (vector<double>) Linear  coefficient of i-th follower's
 * 							  cost. Size of this vector should be equal to n_followers
 * 		capacities			: (vector<double>) Production capacity of each follower.
 * 							  cost. Size of this vector should be equal to n_followers
 *		alpha, beta			: (double, double) Parameters of the demand curve. Written 
 *							  as: Price = alpha - beta*(Total quantity in domestic market) 							  
 *		import_limit		: (double) Maximum net import in the country.
 *							  If no limit, set the value as -1;
 *		export_limit		: (double) Maximum net export in the country.
 *							  If no limit, set the value as -1;
 *		max_tax_perc		: (double) Government decided increase in the shift in costs_lin
 *							  of any player cannot exceed this value
 *		addnlLeadVars		: (unsigned int) Create columns with 0s in it. To handle additional
 *							  dummy leader variables.
 *							  
 * OUTPUTS:					: No output arguments
 * RETURNS:
 * 		NashGame* object	: Points to an object created using "new". A Nash cournot game is played
 * 							  among the followers, for the leader-decided values of import
 * 							  export, caps and taxations on all players. 
 * 							  The total quantity used in the demand equation (See defn of INPUTS: alpha,
 * 							  beta) is the sum of quantity produced by all followers + any import
 * 							  - any export.
 *
 *****************************************************************************************************/
NashGame* createCountry(
		const unsigned int n_followers,
		const vector<double> costs_quad,
		const vector<double> costs_lin,
		const vector<double> capacities,
		const double alpha, const double beta, /// For the demand curve P = a-bQ
		const double import_limit = -1, /// Negative number implies no limit
		const double export_limit = -1,  /// Negative number implies no limit
		const double max_tax_perc = 0.30,
		const unsigned int addnlLeadVars = 0
		);
}; // End of namespace Models {
#endif
