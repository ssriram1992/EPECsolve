#ifndef LCPTOLP_H
#define LCPTOLP_H

#include"epecsolve.h"
#include<iostream>
#include<memory>
#include<gurobi_c++.h>
#include<armadillo>

//using namespace Game;


namespace Game{

arma::vec LPSolve(const arma::sp_mat &A, const arma::vec &b, const arma::vec &c, int &status, bool Positivity=false);
int ConvexHull(const vector<arma::sp_mat*> *Ai, const vector<arma::vec*> *bi, arma::sp_mat &A, arma::vec &b, const arma::sp_mat Acom={}, const arma::vec bcom={});
void compConvSize(arma::sp_mat &A, const unsigned int nFinCons, const unsigned int nFinVar, const vector<arma::sp_mat*> *Ai, 	const vector<arma::vec*> *bi 	);
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
		arma::sp_mat _A={}; arma::vec _b={};	///< Apart from @f$0 \le x \perp Mx+q\ge 0@f$, one needs@f$ Ax\le b@f$ too!
	// Temporary data 
		bool madeRlxdModel{false};	///< Keep track if LCP::RlxdModel is made
		unsigned int nR, nC;
		/// LCP feasible region is a union of polyhedra. Keeps track which of those inequalities are fixed to equality to get the individual polyhedra
		vector<vector<short int>*> *AllPolyhedra, *RelAllPol;
		vector<arma::sp_mat*> *Ai, *Rel_Ai; vector<arma::vec*> *bi, *Rel_bi; 
	   	GRBModel RlxdModel;	///< A gurobi model with all complementarity constraints removed.
	public: 
	// Fudgible data 
		long double bigM {1e5};	///< bigM used to rewrite the LCP as MIP 
		long double eps {1e-5};	///< The threshold, below which a number would be considered to be zero.
	public:
	/** Constructors */
		/// Class has no default constructors
		LCP() = delete;	
		LCP(GRBEnv* env, arma::sp_mat M, arma::vec q, 
				unsigned int LeadStart, unsigned LeadEnd, arma::sp_mat A={}, arma::vec b={}); // Constructor with M,q,leader posn
		LCP(GRBEnv* env, arma::sp_mat M, arma::vec q, 
				perps Compl, arma::sp_mat A={}, arma::vec b={}); // Constructor with M, q, compl pairs
		LCP(GRBEnv* env, const Game::NashGame &N);
	/** Destructor - to delete the objects created with new operator */
		~LCP();
	/** Return data and address */ 
		inline arma::sp_mat getM() {return this->M;}  			///< Read-only access to LCP::M 
		inline arma::sp_mat* getMstar() {return &(this->M);}	///< Reference access to LCP::M 
		inline arma::vec getq() {return this->q;}  				///< Read-only access to LCP::q 
		inline arma::vec* getqstar() {return &(this->q);}		///< Reference access to LCP::q 
		inline unsigned int getLStart(){return LeadStart;} 		///< Read-only access to LCP::LeadStart 
		inline unsigned int getLEnd(){return LeadEnd;}			///< Read-only access to LCP::LeadEnd 
		inline perps getCompl() {return this->Compl;}  			///< Read-only access to LCP::Compl 
		void print(string end="\n");							///< Print a summary of the LCP
		inline unsigned int getNcol() {return this->M.n_cols;};
		inline unsigned int getNrow() {return this->M.n_rows;};
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
		LCP& makeQP(const vector<short int> &Fix, vector<arma::sp_mat*> &custAi, vector<arma::vec*> &custbi, Game::QP_objective &QP_obj, Game::QP_Param &QP);
		LCP& makeQP(Game::QP_objective &QP_obj, Game::QP_Param &QP);
};
};
#endif

