#include<iostream>
#include"func.h"
#include<omp.h>
#include<cmath>
#include<gurobi_c++.h>
#include<armadillo>

using namespace std;


int BinaryArrTest()
	/*
	 * 
	 */
{
	int selecOfTwo[16];
	BinaryArr(selecOfTwo, 16, 1263);
	cout<<endl<<endl;
	for (int i=0; i <16;i++)
		cout<<selecOfTwo[i];
	return 0;
}

int main_LCP()
{
	/* INPUTS */
	int seed = 3;
	unsigned long int N_var = 10;
	long int MentryLim = 10;
	bool VERBOSE{1};

	/* PROGRAM */
	arma::arma_rng::set_seed(seed);
	// arma::arma_rng::set_seed_random();

	/*
	// 0/1 lattice M and q
	arma::sp_mat M(-arma::eye<arma::mat>(N_var,N_var));  // 0/1 lattice
	arma::vec q  = arma::ones<arma::vec>(N_var);
	
	// Random M/q
	arma::sp_mat M(arma::randi<arma::mat>(N_var,N_var, arma::distr_param(-(int)MentryLim, (int)MentryLim))); // Random M
	arma::vec q = arma::randi<arma::vec>(N_var, arma::distr_param(-(int)MentryLim, (int)MentryLim));
	*/
	
	// Nash Cournot M/q
	arma::sp_mat M(N_var, N_var), N(N_var, N_var);
    M	= arma::ones<arma::mat>(N_var, N_var)+arma::eye<arma::mat>(N_var, N_var);
	N = arma::zeros<arma::mat> (N_var, N_var);
	arma::vec q = arma::randi<arma::vec>(N_var, arma::distr_param(1, (int)MentryLim)) - 15;
	

	vector<arma::sp_mat> A{};
	vector<arma::vec> b {};
	vector<arma::vec> sol {};


	// /* COMPUTATION */
	// LCPasLP(M, q, A, b, sol, true, true);
	cout<<"Number of polyhedra considered :"<<LCPasLPTree(M, N, q, A, b, sol, true);
 
	// /* OUTPUT */
	if(VERBOSE) M.impl_print_dense("M");
	if(VERBOSE) q.print("q");
	for (unsigned int i=0; i<A.size();i++)
	{
		 A.at(i).impl_print_dense("A"+to_string(i+1));
		 b.at(i).print("b"+to_string(i+1));
		 sol.at(i).print("sol"+to_string(i+1));
		 cout<<endl<<endl<<endl;
	}
	return 0;
}
// End of #ifndef MAIN_func




int BinaryArr(int *selecOfTwo, unsigned int size, long long unsigned int i)
	/*
	 * Given the size of the output vector "size", and the number "i" in decimal form
	 * converts it into a decimal string of length "size" and stores it in "selecOfTwo"
	 */
{
	for (unsigned int j = size-1; j!=0; j--)
	{
		selecOfTwo[j] = i%2;
		i/=2;
	}
	selecOfTwo[0] = i%2;
	return 0;
}


bool isEmpty(const arma::sp_mat A, const arma::vec b, arma::vec &sol)
	/*
	 * Checks if the polyhedron Ax<=b is empty. If empty, returns true.
	 * If non-empty, returns false and a point "sol" satisfying A*sol<=b
	 */
{
	GRBEnv 		env 		= 		GRBEnv();
	GRBModel 	model 		=		GRBModel(env);

	unsigned int N_vars 			{static_cast<unsigned int>(A.n_cols)};
	unsigned int N_cons 			{static_cast<unsigned int>(A.n_rows)};

	GRBVar 		x[N_vars];

	model.set(GRB_IntParam_OutputFlag, 0);

	if(A.n_rows != b.size()) throw("Inconsistent number of rows in A and b");
	if(A.n_cols != sol.size()) throw("Vector of feasibility proof should have same size as number of cols of A");


	for(unsigned int i=0; i!=N_vars; i++)
		x[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS);

	for(unsigned int i=0; i!=N_cons; i++)
	{
		GRBLinExpr linexp{0};
		for(auto j = A.begin_row(i); j != A.end_row(i); ++j)
			linexp += (*j)*x[j.col()];
		model.addConstr(linexp <= b(i));
	}

	try{
		model.optimize();
	} catch(exception &e){
		cerr<<"Exception: "<<e.what()<<endl;
	}

	int optimstatus;
	optimstatus = model.get(GRB_IntAttr_Status);
	if (optimstatus == GRB_OPTIMAL) 
		for (unsigned int i=0; i!=N_vars; i++)
			sol(i) = x[i].get(GRB_DoubleAttr_X);
	else return true;
	return false;
}


// // // // // // // // // // // // // // // // // // // // // //
// // // // // USE OPENMP TO PARALLELIZE THIS CODE // // // // // 
// // // // // // // // // // // // // // // // // // // // // //

int LCPasLPTree(const arma::sp_mat M, 	// 0 \leq x \perp Mx+Ny+q \geq 0 -> M
		const arma::sp_mat N,		// Matrix for y's coefficients
		const arma::vec q, 			// Corresponding q
		vector<arma::sp_mat> &A, 	// A vector of matrices A_i s.t. \cup_i A_ix \leq b_i gives the 
		vector<arma::vec> &b, 		// set of solutions to the LCP
		vector<arma::vec> &sol,		// In case Gurobiclean is true, returns a feasible point in each of the nonempty polyhedra
		bool cleanup = true 		// If A, b already has elements, this removes them all
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
	int n_Solves{0};
	const unsigned int n_Vars = M.n_cols;
	const unsigned int n_yVars = N.n_cols;
	if (M.n_rows != n_Vars) throw "LCP should be a square system. This has " + to_string(M.n_rows) + " rows"
	   				" and " + to_string(n_Vars) + " columns.";
	if(N.n_rows != n_Vars) throw "Incompatible number of rows in N (0 <= Mx + Ny + q perp x >= 0)";
	if (q.n_rows != n_Vars) throw "q should have same number of rows as M"; 
	long long unsigned int n_Poly = (long long unsigned int)pow(2, n_Vars); // These are the number of polyhedra we might have
	cout<<endl<<endl<<n_Poly<<endl<<endl;

	// Making each of the individual 2^N_vars polyhedra in a tree structure
	for (long long unsigned int i = 0; i< n_Poly;i++)
	{
		int SelecOfTwo[n_Vars];
		bool EmptyFlag{false};
		arma::sp_mat Ai(3*n_Vars, n_Vars + n_yVars);
		arma::vec bi(3*n_Vars, arma::fill::zeros);
		arma::vec soli(n_Vars + n_yVars, arma::fill::zeros);
		unsigned int count = 0; // To keep track of how many constraints are added to Ai and bi
		BinaryArr(SelecOfTwo, n_Vars, i);
		for(unsigned int j=0; j<n_Vars; j++)
		{
			// First both x_i >=0 and [Mx+q]_i >=0 constraints
			Ai(count, j) = -1;
			bi(count++) = 0;
			Ai.row(count) = arma::join_rows(-M.row(j), -N.row(j));
			bi(count++) = q(j);
			if(SelecOfTwo[j]==0) // Then include the x_i <= 0 constraint
			{
				Ai(count, j) = 1;
				bi(count++) = 0;
			}
			else 				// Then include the [Mx+q]_i <= constraint
			{
				Ai.row(count) = arma::join_rows(M.row(j), N.row(j));
				bi(count++) = -q(j); 
			}
			bool t1;
			try
			{
				 t1 = isEmpty(Ai, bi, soli);
			}
			catch(const char* e)
			{
				cerr<<e<<endl;
				exit(1);
			}
			if(t1) // If the polyhdra is already empty with few constraints
			{			
				i += (long long unsigned int)pow(2, n_Vars-j-1);
				i--;
				EmptyFlag = true;
				break;	 
			}
		n_Solves++;
		} 
		bool t1;
		try
		{
			 t1 = isEmpty(Ai, bi, soli);
		}
		catch(const char* e)
		{
			cerr<<e<<endl;
			exit(1);
		}
		if (!EmptyFlag && !t1)
		{
			 A.push_back(Ai);
			 b.push_back(bi);
			 sol.push_back(soli);
			 n_Solves++;;
		}
	}
	return n_Solves;
}

