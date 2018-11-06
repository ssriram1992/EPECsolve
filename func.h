#ifndef FUNC_H
#define FUNC_H

#include<iostream>
#include<ctime>
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
int LCPasLPTree(const arma::sp_mat M, const arma::sp_mat N,	const arma::vec q,	vector<arma::sp_mat> &A,vector<arma::vec> &b, vector<arma::vec> &sol,	bool cleanup);
int LCPasLP(const arma::sp_mat M,	const arma::vec q,	vector<arma::sp_mat> &A,vector<arma::vec> &b, vector<arma::vec> &sol,	bool cleanup , 	bool Gurobiclean );
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
 * \max_y \frac{1}{2}y^TQy + c^Ty + (Cx)^T y
 * Subject to
 * Ax + By <= b
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
		}
		unsigned int KKT(arma::sp_mat& M, arma::sp_mat& N, arma::vec& q) const;
	public: // Set some data
		QP_Param& set(arma::sp_mat Q, arma::sp_mat C, arma::sp_mat A, arma::sp_mat B, arma::vec c, arma::vec b); // Copy data into this
		QP_Param& setMove(arma::sp_mat Q, arma::sp_mat C, arma::sp_mat A, arma::sp_mat B, arma::vec c, arma::vec b); // Move data into this
	public: // Return some of the data as a copy
		arma::sp_mat getQ() const; arma::sp_mat getC() const; arma::sp_mat getA() const;
		arma::sp_mat getB() const; arma::vec getc() const; arma::vec getb() const;
		unsigned int getNx() const; unsigned int getNy() const;
	public: // Other methods
		bool is_Playable(QP_Param P const) const;
};



class NashGame
{
	public: // Variables
		unsigned int Nplayers;
		vector<QP_Param*> Players;
		arma::mat MarketClearing;
		// In the vector of variables of all players,
		// which position does the variable corrresponding to this player starts.
		vector<unsigned int> primal_position; 
		vector<unsigned int> dual_position; 
		vecor<unsigned int> MC_dual_position;
	public: // Constructors
		NashGame(unsigned int Nplayers):Nplayers{Nplayers} 
		{
			Players.resize(this->Nplayers); 
			primal_position.resize(this->Nplayers);
			dual_position.resize(this->Nplayers);
		}
	public: // Members
		unsigned int FormulateLCP(arma::sp_mat &M, arma::vec &q) const;
};

void MPEC(NashGame N, arma::sp_mat Q, QP_Param &P);







#endif
