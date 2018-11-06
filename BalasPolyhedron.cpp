#include<iostream>
#include"func.h"
#include<ctime>
#include<cstdlib>
#include<gurobi_c++.h>
#include<armadillo>
// Define this as 0 if you don't want the equations fo each polyhedron printed out.
#define VERBOSE 0

#include<exception>

using namespace std;


// Function prototypes
// GRBModel& PolyUnion(GRBModel &model, GRBVar **&x, GRBVar *&xMain, GRBVar *&delta, 
		// const vector<arma::sp_mat> A, const vector<arma::vec> b);
// 
// GRBModel& PolyUnion(GRBModel &model, GRBVar **&x, GRBVar *&xMain, GRBVar *&delta, 
		// const vector<arma::mat> A, const vector<arma::vec> b);

// These bounds exist on all variables. 
// These ensure that the whole problem is never unbounded
constexpr double LB = -100;
constexpr double UB = -LB;

#ifndef MAIN_func
#define MAIN_func
int main_Balas()
{
	// DECLARATIONS
	constexpr int N_poly 						= 		3; // Union over this many polyhedra
	constexpr int N_vars 						= 		4; // The number of variables
	constexpr int max_mat_size 					= 		15; // each polyhedra will not have more than this many facets
	int optimstatus;										// To store if the problem was infeasible or unbounded or has optimal solution

	vector<unsigned int> 						sizes(N_poly); // Stores the number of constraints for each polyhedron
	arma::vec 									objective;
	vector<arma::mat> 							A(N_poly);
	vector<arma::sp_mat>						A2(N_poly);
	vector<arma::vec> 							b(N_poly);

	GRBEnv		env 							=		GRBEnv();
	GRBModel	model 							=		GRBModel(env);

	GRBVar										**x = nullptr;
	GRBVar										*xMain = nullptr;
	GRBVar 										*delta = nullptr;

	// INITIALIZATIONS
	srand(6);						// Seed for random number generation. Change this to get a different realization.

	// Initializing the size of each polyhedra
	for (int i=0; i!=N_poly; i++)
		sizes[i]=rand()%max_mat_size;

	// Initializing the constraint matrices and RHS
	for (int i=0; i!=N_poly; i++)
	{
		A2[i] = arma::sprandn<arma::sp_mat>(sizes[i], N_vars, 0.4)*10; 	// This is the sparse matrix
		A[i] = arma::mat(A2[i]);										// This is the dense matrix
		// Initialize the RHS such that each polyhedron is non-empty for sure
		// To maintain simplicity to avoid handling the case where the whole problem is infeasible
		b[i] = arma::vec(A[i]*arma::randi<arma::vec>(N_vars, arma::distr_param(1,5))) + 1;
	}

	// Initializing the Objective function
	objective = arma::randn(N_vars);
	objective.fill(1.0);

	// Print the constraint matrices and RHS
	for (int i=0; i!=N_poly; i++) 
	{
		if(VERBOSE) A[i].print("A element "+to_string(i)+": ");
		if(VERBOSE) b[i].print("b element "+to_string(i)+": ");
	}
	if(VERBOSE) objective.print("Objective:");


	// Only one of the two lines should be uncommented to prevent any erroneous behaviour
	// PolyUnion(model, x, xMain, delta, A, b); // Uncomment this line to solve the model with dense matrix A
	PolyUnion(model, x, xMain, delta, A2, b); // Uncomment this line to solve the model with sparse matrix A2

	// Populate random objective
	GRBLinExpr obj{0};
	for(unsigned int k=0; k!=N_vars; k++)
		obj += objective(k)*xMain[k];
	model.setObjective(obj, GRB_MINIMIZE);

	// Gurobi does the magic
	try{
		cout<<"Exception block try"<<endl;
	model.optimize();
	} catch(exception &e){
		cout<<"Exception block catch";
		cerr<<"Exception: "<<e.what()<<endl;
	}
	optimstatus = model.get(GRB_IntAttr_Status);

	if(optimstatus == GRB_INFEASIBLE)
		 cout<<"Infeasible problem. Probably each of the mentioned polyhedron is empty";
	else if(optimstatus == GRB_UNBOUNDED)
		cout<<"Unbounded objective. ";
	else if(optimstatus == GRB_OPTIMAL)
	{
		 cout<<"Optimal solution found\n";
		// Retrieving the optimal solution.
		double soln[N_vars]{};
		for(int k=0; k!=N_vars;k++)
			soln[k] = xMain[k].get(GRB_DoubleAttr_X);
		// Printing the optimal solution.
		cout<<"Solution:"<<endl;
		for(int k=0; k!=N_vars;k++)
			cout<<soln[k]<<"\t";
	}
	else
		cout<<"Infeasible dual. Primal may be feasible or unbounded"<<endl;

	// Freeing up the memory
	delete[] xMain;
	delete[] delta;
	for(int i=0; i!=N_poly; i++)
		delete x[i];
	delete[] x;
	cout<<endl;

	return 0;
}
// End of MAIN_func
#endif

template <class T>
vector<unsigned int> errChk_PolyUnion(
		const vector<T>		A,				// Vector containing the LHS 'A' in Ax <= b defining each polyhedron
		const vector<arma::vec>		b		// Vector containing the RHS 'b' in Ax <= b defining each polyhedron 
		)
{
	const unsigned int 							N_poly = A.size();						// Define number of polyhedron
	const unsigned int 							N_vars = A[0].n_cols;					// Number of variables
	vector<unsigned int>						sizes{}; 

	// Check the following:
	// 			There is a non-empty union of polyhedra
	// 			Number of LHS and RHS are same 
	// 			Each A has same number of variables.
	// 			Each A and corresponding b have same number of constraints.
	if (A.size() == 0) 					throw "Empty vector of polyhedra given!";	// There should be at least 1 polyhedron to consider
	if (A.size() != b.size()) 			throw "Inconsistent number of LHS and RHS for polyhedra"; // Vector and A should have same size
	for(unsigned int i=0; i!=N_poly; i++)
	{
		if (A[i].n_cols != N_vars) 		throw "Inconsistent number of variables in the polyhedra " + to_string(i);
		if (A[i].n_rows != b[i].n_rows) throw "Inconsistent number of rows in LHS and RHS of polyhedra " + to_string(i);
		sizes.push_back(A[i].n_rows);
	} 	
	return sizes;
}


// Function definition for dense representation
GRBModel& PolyUnion(
		// Output arguments
		GRBModel 				&model, 			// Gurobi model object where the variables and constraints will be added
		GRBVar 					**&x, 				// Object where the auxilary x variables for each polyhedron exist
		GRBVar 					*&xMain, 			// Object for the main x variables
		GRBVar 					*&delta, 			// Auxiliary delta variables
		// Input arguments
		const vector<arma::mat>		A,				// Vector containing the LHS 'A' in Ax <= b defining each polyhedron
		const vector<arma::vec>		b				// Vector containing the RHS 'b' in Ax <= b defining each polyhedron 
		)
{
	GRBLinExpr 									lincons;
	vector<unsigned int>						sizes{};
	const unsigned int 							N_poly = A.size();						// Define number of polyhedron
	const unsigned int 							N_vars = A[0].n_cols;					// Number of variables

	sizes = errChk_PolyUnion<arma::mat>(A, b);
	
	// Creating the Gurobi variables and constraints
	xMain 			= 			new GRBVar[N_vars];
	delta 			= 			new GRBVar[N_poly];
	x 				= 			new GRBVar*[N_poly];

	for(unsigned int i=0; i!=N_poly; i++)
		x[i] = new GRBVar[N_vars];

	for (unsigned int i=0; i!=N_poly; i++)					// Iterating over each polyhedron
	{
		for(unsigned int k=0; k!=N_vars; k++)				// Adding variables to the model
			x[i][k] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "x_"+to_string(i)+"_"+to_string(k));

		delta[i] = model.addVar(0, 1, 0, GRB_CONTINUOUS, "delta_"+to_string(i));

		for(unsigned int j=0; j!=sizes[i]; j++)		// Iterating over each constraint and adding them
		{
			lincons = 0;
			for(unsigned int k=0; k!=N_vars; k++)			// Iterating over each variable in a constraint
				lincons+=A[i](j, k)*x[i][k];
			model.addConstr(lincons <= delta[i]*b[i](j), "Constr_"+to_string(i)+"_"+to_string(j));
		}
	}

	// Adding the delta constraints
	lincons = 0;
	for (unsigned int i=0; i!=N_poly; i++)
		lincons += delta[i];
	model.addConstr(lincons == 1, "delta");

	// Adding the constraints linking the actual variables and polyhedron-specific variables
	for (unsigned int k=0; k!= N_vars; k++)					// Adding the actual variables
	{
		xMain[k] = model.addVar(LB, UB, 1, GRB_CONTINUOUS, "xMain_"+to_string(k));
		lincons = GRBLinExpr{0};
		for (unsigned int i=0; i!=N_poly; i++)
			lincons+=x[i][k];
		lincons -= xMain[k];
		model.addConstr(lincons==0, "Projection");
	}
	return model;
}



// Function definition for sparse representation
GRBModel& PolyUnion(
		// Output arguments
		GRBModel 				&model, 			// Gurobi model object where the variables and constraints will be added
		GRBVar 					**&x, 				// Object where the auxilary x variables for each polyhedron exist
		GRBVar 					*&xMain, 			// Object for the main x variables
		GRBVar 					*&delta, 			// Auxiliary delta variables
		// Input arguments
		const vector<arma::sp_mat>	A,				// Vector containing the LHS 'A' in Ax <= b defining each polyhedron
		const vector<arma::vec>		b				// Vector containing the RHS 'b' in Ax <= b defining each polyhedron 
		)
{
	GRBLinExpr 									lincons;
	vector<unsigned int>						sizes{};
	const unsigned int 							N_poly = A.size();						// Define number of polyhedron
	const unsigned int 							N_vars = A[0].n_cols;					// Number of variables

	sizes = errChk_PolyUnion<arma::sp_mat>(A, b);
	
	// Creating the Gurobi variables and constraints
	xMain 			= 			new GRBVar[N_vars];
	delta 			= 			new GRBVar[N_poly];
	x 				= 			new GRBVar*[N_poly];
	for(unsigned int i=0; i!=N_poly; i++)
		x[i] = new GRBVar[N_vars];

	for (unsigned int i=0; i!=N_poly; i++)					// Iterating over each polyhedron
	{
		for(unsigned int k=0; k!=N_vars; k++)				// Adding variables to the model
			x[i][k] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "x_"+to_string(i)+"_"+to_string(k));

		delta[i] = model.addVar(0, 1, 0, GRB_CONTINUOUS, "delta_"+to_string(i));

		for(unsigned int j=0; j!=sizes[i]; j++)		// Iterating over each constraint and adding them
		{
			lincons = 0;
			for(auto k=A[i].begin_row(j); k!=A[i].end_row(j); ++k)
				lincons += (*k)*x[i][k.col()] ;
			model.addConstr(lincons <= delta[i]*b[i](j), "Constr_"+to_string(i)+"_"+to_string(j));
		}
	}

	// Adding the delta constraints
	lincons = 0;
	for (unsigned int i=0; i!=N_poly; i++)
		lincons += delta[i];
	model.addConstr(lincons == 1, "delta");

	// Adding the constraints linking the actual variables and polyhedron-specific variables
	for (unsigned int k=0; k!= N_vars; k++)					// Adding the actual variables
	{
		xMain[k] = model.addVar(LB, UB, 0, GRB_CONTINUOUS, "xMain_"+to_string(k));
		lincons = GRBLinExpr{0};
		for (unsigned int i=0; i!=N_poly; i++)
			lincons+=x[i][k];
		lincons -= xMain[k];
		model.addConstr(lincons==0, "Projection");
	}

	return model;
}



int PolyUnion(
		// Input arguments
		const vector<arma::sp_mat>	Ai,				// Vector containing the LHS 'A' in Ax <= b defining each polyhedron
		const vector<arma::vec>		bi,				// Vector containing the RHS 'b' in Ax <= b defining each polyhedron 
		// Output arguments
		arma::sp_mat& A,
		arma::vec &b,
		bool Reduce
		)
{
	vector<unsigned int>						sizes{};
	const unsigned int 							N_poly = Ai.size();						// Define number of polyhedron
	const unsigned int 							N_vars = Ai[0].n_cols;					// Number of variables

	const unsigned int 							FN_vars = N_poly*N_vars + N_vars + N_poly;
	unsigned int 								FN_cons{2+N_vars+N_poly}; // Delta sum to 1, x^i sum to x and each delta_i >= 0.
	for(auto i:Ai)												   // Adding the number of constraints in each polyhedron description.
		FN_cons += i.n_rows;


	sizes = errChk_PolyUnion<arma::sp_mat>(Ai, bi);
	A.set_size(FN_cons, FN_vars); // Setting the size of A
	b.set_size(FN_cons); 		  // Setting the size of B
	b.fill(0);

	
	// Populating the A, b matrices
	unsigned int cons_count{0}, var_count{0};
	for(unsigned int i=0; i<N_poly; i++)
	{
		cout<<i<<endl;
		// A_ix^i \leq \delta_i b^i constraint in Conforti, Cornuejols, Zambelli (4.31)
		// Add Ai(i) as the appropriate submatrix of the new A
		A.submat(cons_count, var_count, 
				cons_count+bi.at(i).n_rows-1, var_count+N_vars-1) = Ai.at(i);
		// Each row also has +\delta_ib^i \leq 0
		A.submat(cons_count, N_poly*N_vars + N_vars + i,
			cons_count+bi.at(i).n_rows-1, N_poly*N_vars + N_vars + i) = -bi.at(i);
		// \sum x^i = x constraint - adding the term corresponding to this polyhedra
		A.submat(FN_cons - 2- N_poly - N_vars, var_count,
				FN_cons - 2 - N_poly -1, var_count+N_vars-1) = arma::speye<arma::sp_mat>(N_vars, N_vars);
		// Updating the constraint and variable count
		cons_count += bi.at(i).n_rows;
		var_count += N_vars;
		cout<<i<<" over"<<endl;
	}
	A.submat(FN_cons - 2- N_poly - N_vars, N_poly*N_vars,
			FN_cons - 2 - N_poly -1, N_poly*N_vars+N_vars-1) = arma::speye<arma::sp_mat>(N_vars, N_vars);
	A.submat(FN_cons-2-N_poly,N_poly*N_vars + N_vars,
				FN_cons-2-1, N_poly*N_vars + N_vars + N_poly-1) = -arma::speye<arma::sp_mat>(N_poly, N_poly);
	A.submat(FN_cons-2, N_poly*N_vars+N_vars,
			FN_cons-2,  N_poly*N_vars+N_vars+N_poly-1) = arma::ones<arma::mat>(1, N_poly);
	b[FN_cons-2] = 1;
	A.submat(FN_cons-1, N_poly*N_vars+N_vars,
			FN_cons-1,  N_poly*N_vars+N_vars+N_poly-1) = -arma::ones<arma::mat>(1, N_poly);
	b[FN_cons-1] = -1;

	return 0;
}


/*

int PolyUnion(
		// Input arguments
		const vector<arma::mat>	Ai,				// Vector containing the LHS 'A' in Ax <= b defining each polyhedron
		const vector<arma::vec>		bi,				// Vector containing the RHS 'b' in Ax <= b defining each polyhedron 
		// Output arguments
		arma::mat A,
		arma::vec b
		)
{
	 return 0;
}

*/



vector<unsigned int> makeCompactPolyhedron(const arma::sp_mat A, const arma::vec b, arma::sp_mat &Anew, arma::vec &bnew)
{
	GRBEnv env = GRBEnv();
	GRBModel model = GRBModel(env);
	model.set(GRB_IntParam_DualReductions, 0);
		// ("DualReductions",0);
	
	GRBVar x[A.n_cols];

	GRBConstr Constraints[A.n_rows];
	GRBLinExpr LinExpr[A.n_rows], lin{0};
	
	vector<unsigned int> remainConst{};

	double obj;
	// Add all variables - objective does not matter for now.
	for(unsigned int i = 0; i<A.n_cols; i++)
		x[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS);
	for(unsigned int i=0; i<A.n_rows;i++)
	{
		for(auto j=A.begin_row(i);j!=A.end_row(i);++j)
			LinExpr[i] += (*j)*x[j.col()];
		Constraints[i] = model.addConstr(LinExpr[i], GRB_LESS_EQUAL, b[i]);
	}
	model.setObjective(lin);
	model.optimize();
	// model.set(GRB_IntParam_OutputFlag, 0);
	auto optimstatus = model.get(GRB_IntAttr_Status);
	if(optimstatus == GRB_INFEASIBLE || optimstatus== GRB_INF_OR_UNBD)
	{
		cout<<"Polyhedron that was cleaned is infeasible already!";
		return remainConst;
	}
	// Remove constraint by constraint and see if it is required
	for(unsigned int i=0; i<A.n_rows;i++)
	{
		model.setObjective(LinExpr[i], GRB_MAXIMIZE);
		model.remove(Constraints[i]);
		try{ model.optimize();}
		catch(exception &e) { cerr<<"Exception: "<<e.what()<<endl; }
		auto optimstatus = model.get(GRB_IntAttr_Status);
		if(optimstatus != GRB_UNBOUNDED && optimstatus != 4) // 4 is for infeasible or unbounded
		{
			obj = model.get(GRB_DoubleAttr_ObjVal);
			if(obj > b[i])
			{
				remainConst.push_back(i);
				model.addConstr(LinExpr[i], GRB_LESS_EQUAL, b[i]);
			}
		}
		else
		{
			remainConst.push_back(i);
			model.addConstr(LinExpr[i], GRB_LESS_EQUAL, b[i]);
		}
	}
	arma::ivec remains(remainConst.size());
	Anew = arma::zeros<arma::mat>(remainConst.size(), A.n_cols);
	bnew = arma::zeros<arma::vec>(remainConst.size());
	for(unsigned int i = 0; i<remainConst.size(); i++)
	{
		Anew.row(i) = A.row(remainConst.at(i));
		bnew.row(i) = b(remainConst.at(i));
		remains[i] = remainConst.at(i);
	}
	return remainConst;
}
