#include<iostream>
#include"func.h"
#include<omp.h>
#include<cmath>
#include<gurobi_c++.h>
#include<armadillo>


using namespace std;


int LCPasLP(const arma::sp_mat M, 	// 0 \leq x \perp Mx+q \geq 0 -> M
		const arma::vec q, 			// Corresponding q
		vector<arma::sp_mat> &A, 	// A vector of matrices A_i s.t. \cup_i A_ix \leq b_i gives the 
		vector<arma::vec> &b, 		// set of solutions to the LCP
		vector<arma::vec> &sol,		// In case Gurobiclean is true, returns a feasible point in each of the nonempty polyhedra
		bool cleanup = true, 		// If A, b already has elements, this removes them all
		bool Gurobiclean = true		// Encouraged to keep this true. Throws away any empty polyhedron while making them.
		)
/*
	Creates the matrices A_i and b_i in A_ix \leq b_i notation where each of them is a polyhedron 
	whose union is the solution to the original LP
*/
{
	// Erase the vectors A and b if cleanup is given
	if(cleanup)
	{
		A.erase(A.begin(), A.end());
		b.erase(b.begin(), b.end());
		sol.erase(sol.begin(), sol.end());
	} 
	// Declaration
	const unsigned int n_Vars = M.n_cols;
	if (M.n_rows != n_Vars) throw "LCP should be a square system. This has " + to_string(M.n_rows) + " rows"
	   				" and " + to_string(n_Vars) + " columns.";
	if (q.n_rows != n_Vars) throw "q should have same number of rows as M"; 
	long long unsigned int n_Poly = (long long unsigned int)pow(2, n_Vars); // These are the number of polyhedra we might have
	cout<<endl<<endl<<n_Poly<<endl<<endl;

	// Making each of the individual 2^N_vars polyhedra.
	for (long long unsigned int i = 0; i< n_Poly;i++)
	{
		int SelecOfTwo[n_Vars];
		arma::sp_mat Ai(3*n_Vars, n_Vars);
		arma::vec bi(3*n_Vars);
		arma::vec soli(n_Vars);
		unsigned int count = 0; // To keep track of how many constraints are added to Ai and bi
		BinaryArr(SelecOfTwo, n_Vars, i);
		for(unsigned int j=0; j<n_Vars; j++)
		{
			// First both x_i >=0 and [Mx+q]_i >=0 constraints
			Ai(count, j) = -1;
			bi(count++) = 0;
			Ai.row(count) = -M.row(j);
			bi(count++) = q(j);
			if(SelecOfTwo[j]==0) // Then include the x_i <= 0 constraint
			{
				Ai(count, j) = 1;
				bi(count++) = 0;
			}
			else 				// Then include the [Mx+q]_i <= constraint
			{
				Ai.row(count) = M.row(j);
				bi(count++) = -q(j); 
			}
		}
		if(Gurobiclean) // If GurobiClean option is set to true, then only those polyhedra that are not empty!
		{
			if (!isEmpty(Ai, bi, soli))
			{
				 A.push_back(Ai);
				 b.push_back(bi);
				 sol.push_back(soli);
			}
		}
		else
		{
			 A.push_back(Ai);
			 b.push_back(bi);
		}
	}
	return 0;
}
