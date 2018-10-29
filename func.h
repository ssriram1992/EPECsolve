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


/************************************************************************************************************************/
/* 																														*/
/******************************************			FROM LCPTOLP.CPP			*****************************************/ 
/* 																														*/
/************************************************************************************************************************/
int LCPasLPTree(const arma::sp_mat M, const arma::sp_mat N,	const arma::vec q,	vector<arma::sp_mat> &A,vector<arma::vec> &b, vector<arma::vec> &sol,	bool cleanup);
int LCPasLP(const arma::sp_mat M,	const arma::vec q,	vector<arma::sp_mat> &A,vector<arma::vec> &b, vector<arma::vec> &sol,	bool cleanup , 	bool Gurobiclean );
int BinaryArr(int *selecOfTwo, unsigned int size, long long unsigned int i);
bool isEmpty(const arma::sp_mat A, const arma::vec b, arma::vec &sol);

#endif
