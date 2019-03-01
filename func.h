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

/************************************************************************************************************************/
/* 																														*/
/******************************************		FROM BALASPOLYHEDRON.CPP		*****************************************/ 
/* 																														*/
/************************************************************************************************************************/
// Returns a Gurobi model which can optimize over the convex hull of the 
// union of polyhedra described in A and b where A and b are dense
GRBModel& PolyUnion(GRBModel &model, GRBVar **&x, GRBVar *&xMain, GRBVar *&delta, 
		const vector<arma::sp_mat> A, const vector<arma::vec> b);

// Returns a Gurobi model which can optimize over the convex hull of the 
// union of polyhedra described in A and b where A and b are sparse
GRBModel& PolyUnion(GRBModel &model, GRBVar **&x, GRBVar *&xMain, GRBVar *&delta, 
		const vector<arma::mat> A, const vector<arma::vec> b);

int PolyUnion(const vector<arma::sp_mat> Ai, const vector<arma::vec> bi, arma::sp_mat& A, arma::vec &b, bool Reduce=false);

vector<unsigned int> makeCompactPolyhedron(const arma::sp_mat A, const arma::vec b, arma::sp_mat &Anew, arma::vec &bnew);

/************************************************************************************************************************/
/* 																														*/
/******************************************			FROM LCPTOLP.CPP			*****************************************/ 
/* 																														*/
/************************************************************************************************************************/

typedef vector<pair<unsigned int, unsigned int>> perps  ;
ostream& operator<<(ostream& ost, perps C);
bool operator < (vector<int> Fix1, vector<int> Fix2);
bool operator == (vector<int> Fix1, vector<int> Fix2);
template <class T> ostream& operator<<(ostream& ost, vector<T> v);
template <class T, class S> ostream& operator<<(ostream& ost, pair<T,S> p);


class LCP{
	/**
	* A class to handle linear complementarity problems (LCP)
	* especially as MIPs with bigM constraints
	*/
	private:
	// Essential data
		GRBEnv* env;
		arma::sp_mat M; arma::vec q; perps Compl; /// Compl stores data in <Eqn, Var> form.
		unsigned int LeadStart, LeadEnd, nLeader;
	// Temporary data
		bool madeRlxdModel;
		unsigned int nR, nC;
		vector<vector<int>*>* AllPolyhedra;
	   	GRBModel RlxdModel;
	public: 
	// Fudgible data
		long double bigM;
		long double eps;
	public:
	/** Constructors */
		LCP() = delete;	/// Class has no default constructors
		LCP(GRBEnv* env, arma::sp_mat M, arma::vec q, unsigned int LeadStart, unsigned LeadEnd); // Constructor with M,q,leader posn
		LCP(GRBEnv* env, arma::sp_mat M, arma::vec q, perps Compl); // Constructor with M, q, compl pairs
	/** Return data and address */
		arma::sp_mat getM() {return this->M;}  arma::sp_mat* getMstar() {return &(this->M);}
		arma::vec getq() {return this->q;}  arma::vec* getqstar() {return &(this->q);}
		unsigned int getLStart(){return LeadStart;} unsigned int getLEnd(){return LeadEnd;}
		perps getCompl() {return this->Compl;}  
		friend ostream& operator<<(ostream& ost, const LCP L);
	/* Member functions */
	private:
		bool errorCheck(bool throwErr=true) const;
		void defConst(GRBEnv* env);
		int makeRelaxed();
		template<class T> bool isZero(const T val) const { return (val>-eps && val < eps);}
		vector<int>* solEncode(GRBModel *model)const;
		vector<int>* solEncode(const arma::vec &z, const arma::vec &x)const;
		void branch(int loc, const vector<int> *Fixes);
		vector<int>* anyBranch(const vector<vector<int>*>* vecOfFixes, vector<int>* Fix) const;
		int BranchLoc(GRBModel* m, vector<int>* Fix);
		int BranchProcLoc(vector<int>* Fix, vector<int> *Leaf);
	public:
		GRBModel* LCPasMIP(vector<unsigned int> FixEq={}, vector<unsigned int> FixVar={}, bool solve=false);
		GRBModel* LCPasMIP(vector<int> Fixes, bool solve);
		GRBModel* LCP_Polyhed_fixed(vector<unsigned int> FixEq={}, vector<unsigned int> FixVar={});
		GRBModel* LCP_Polyhed_fixed(arma::Col<int> FixEq, arma::Col<int> FixVar);
		bool extractSols(GRBModel* model, arma::vec &z, arma::vec &x, bool extractZ = false) const;

		vector<vector<int>*> *BranchAndPrune ();
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

/************************************************************************************************************************/
/* 																														*/
/******************************************			FROM GAMES.CPP				*****************************************/ 
/* 																														*/
/************************************************************************************************************************/

class QP_Param
/* 
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
		QP_Param(arma::sp_mat Q, arma::sp_mat C, arma::sp_mat A, arma::sp_mat B, arma::vec c, arma::vec b)
		{
			this->set(Q, C, A, B, c, b);
			this->size();
		}
	public: // Set some data
		QP_Param& set(arma::sp_mat Q, arma::sp_mat C, arma::sp_mat A, arma::sp_mat B, arma::vec c, arma::vec b); // Copy data into this
		QP_Param& setMove(arma::sp_mat Q, arma::sp_mat C, arma::sp_mat A, arma::sp_mat B, arma::vec c, arma::vec b); // Move data into this
	public: // Return some of the data as a copy
		arma::sp_mat getQ() const; arma::sp_mat getC() const; arma::sp_mat getA() const;
		arma::sp_mat getB() const; arma::vec getc() const; arma::vec getb() const;
		unsigned int getNx() const; unsigned int getNy() const;
	public: // Other methods
		unsigned int KKT(arma::sp_mat& M, arma::sp_mat& N, arma::vec& q) const;
		bool is_Playable(const QP_Param P) const;
};

class NashGame
{
	public: // Variables
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
		NashGame(vector<QP_Param*> Players, arma::sp_mat MC, arma::vec MCRHS, unsigned int n_LeadVar=0);
		NashGame(unsigned int Nplayers, unsigned int n_LeadVar=0):Nplayers{Nplayers} , n_LeadVar{n_LeadVar}
		{
			Players.resize(this->Nplayers); 
			primal_position.resize(this->Nplayers);
			dual_position.resize(this->Nplayers);
		}
	public: // Members
		unsigned int FormulateLCP(arma::sp_mat &M, arma::vec &q, perps &Compl) const;
};

void MPEC(NashGame N, arma::sp_mat Q, QP_Param &P);
ostream& operator<< (ostream& os, const QP_Param &Q);
ostream& operator<< (ostream& os, const NashGame N);

#endif
