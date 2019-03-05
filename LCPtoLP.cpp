#include<iostream>
#include<string>
#include"func.h"
#include<gurobi_c++.h>
#include<armadillo>

using namespace std;


int BinaryArr(int *selecOfTwo, unsigned int size, long long unsigned int i)
/**
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

void LCP::defConst(GRBEnv* env)
{
	AllPolyhedra = new vector<vector<int>*> {};
	Ai = new vector<arma::sp_mat *>{}; bi = new vector<arma::vec *>{};
	if(!VERBOSE) this->RlxdModel.set(GRB_IntParam_OutputFlag,0);
	this->env = env;  this->madeRlxdModel = false; this->bigM = 1e5; this->eps = 1e-5;
	this->nR = this->M.n_rows; this->nC = this->M.n_cols;
}


LCP::LCP(GRBEnv* env, arma::sp_mat M, arma::vec q, perps Compl, arma::sp_mat A, arma::vec b):M{M}, q{q}, _A{A}, _b{b}, RlxdModel(*env) /// Constructor with M, q, compl pairs
{
	defConst(env);
	this->Compl = Compl;
	sort(Compl.begin(), Compl.end(), 
		[](pair<unsigned int, unsigned int>a, pair<unsigned int, unsigned int> b) 
			{return a.first < b.first;}
		);
	for(auto p:Compl)
		if(p.first!=p.second) 
		{
			this->LeadStart = p.first;		 this->LeadEnd = p.second - 1;
			this->nLeader = this->LeadEnd-this->LeadStart + 1;  
			this->nLeader = this->nLeader>0? this->nLeader:0;
			break;
		}
}

LCP::LCP(GRBEnv* env, arma::sp_mat M, arma::vec q, unsigned int LeadStart, unsigned LeadEnd, arma::sp_mat A, arma::vec b):M{M}, q{q}, _A{A}, _b{b}, RlxdModel(*env) /// Constructor with M,q,leader posn
{
	defConst(env);
	this->LeadStart = LeadStart; this->LeadEnd = LeadEnd;
	this->nLeader = this->LeadEnd-this->LeadStart + 1;  
	this->nLeader = this->nLeader>0? this->nLeader:0;
	for(unsigned int i=0;i <M.n_rows;i++)
	{
		unsigned int count = i<LeadStart?i:i+nLeader;
		Compl.push_back({i,count});
	}
}

int LCP::makeRelaxed()
{
	try
	{
		if(this->madeRlxdModel) return 0;
		GRBVar x[nC], z[nR];
		for(unsigned int i=0;i <nC;i++) x[i] = RlxdModel.addVar(0, GRB_INFINITY, 1, GRB_CONTINUOUS, "x_"+to_string(i));
		for(unsigned int i=0;i <nR;i++) z[i] = RlxdModel.addVar(0, GRB_INFINITY, 1, GRB_CONTINUOUS, "z_"+to_string(i));
		for(unsigned int i=0;i <nR;i++)
		{
			GRBLinExpr expr = 0;
			for(auto v=M.begin_row(i); v!=M.end_row(i); ++v)
				expr += (*v) * x[v.col()];
			expr += q(i);
			RlxdModel.addConstr(expr, GRB_EQUAL, z[i]);
		} 
		// If Ax \leq b constraints are there, they should be included too!
		if(this->_A.n_nonzero != 0 || this->_b.n_rows!=0)
		{ 
			if(_A.n_cols != nC || _A.n_rows != _b.n_rows) throw "A and b are incompatible! Thrown from makeRelaxed()";
			for(unsigned int i=0;i<_A.n_rows;i++)
			{
				GRBLinExpr expr = 0;
				for(auto a=_A.begin_row(i); a!=M.end_row(i); ++a)
					expr += (*a) * x[a.col()];
				RlxdModel.addConstr(expr, GRB_LESS_EQUAL, _b(i));
			}
		}
		RlxdModel.update();
		this->madeRlxdModel = true;
	}
	catch(const char* e) { cout<<e<<endl; }
	catch(string e) { cout<<"String: "<<e<<endl; }
	catch(exception &e) { cout<<"Exception: "<<e.what()<<endl; }
	catch(GRBException &e){cout<<"GRBException: "<<e.getErrorCode()<<"; "<<e.getMessage()<<endl;}
	if(!this->madeRlxdModel) throw "STOP!!! STOP!!! STOP!!!";
	return 0;
}

GRBModel* LCP::LCP_Polyhed_fixed(
		vector<unsigned int> FixEq,  		// If non zero, equality imposed on variable
		vector<unsigned int> FixVar  		// If non zero, equality imposed on equation
		)			
/**
 * Returs a model 
 * The returned model has constraints
 * corresponding to the non-zero elements of FixEq set to equality
 * and variables corresponding to the non-zero
 * elements of FixVar set to equality (=0)
 */
{
	makeRelaxed();
	GRBModel* model = new GRBModel(this->RlxdModel);
	for(auto i:FixEq)
	{
		if(i>=nR) throw "Element in FixEq is greater than nC";
		model->getVarByName("z_"+to_string(i)).set(GRB_DoubleAttr_UB,0);
	}
	for(auto i:FixVar)
	{
		if(i>=nC) throw "Element in FixEq is greater than nC";
		model->getVarByName("z_"+to_string(i)).set(GRB_DoubleAttr_UB,0);
	}
	return model;
}

GRBModel* LCP::LCP_Polyhed_fixed(
		arma::Col<int> FixEq,  		// If non zero, equality imposed on variable
		arma::Col<int> FixVar  		// If non zero, equality imposed on equation
		)			
/**
 * Returs a model created from a given model
 * The returned model has constraints
 * corresponding to the non-zero elements of FixEq set to equality
 * and variables corresponding to the non-zero
 * elements of FixVar set to equality (=0)
 */
{
	makeRelaxed();
	GRBModel* model = new GRBModel(this->RlxdModel);
	for(unsigned int i=0;i<nC;i++)
		if(FixVar[i]) 
			model->getVarByName("x_"+to_string(i)).set(GRB_DoubleAttr_UB,0);
	for(unsigned int i=0;i<nR;i++)
		if(FixEq[i]) 
			model->getVarByName("z_"+to_string(i)).set(GRB_DoubleAttr_UB,0);
	model->update();
	return model;
}

GRBModel* LCP::LCPasMIP(vector<int> Fixes, bool solve)
{
	if(Fixes.size()!=this->nR) throw "Bad size for Fixes in LCP::LCPasMIP";
	vector<unsigned int> FixVar, FixEq; 
	for(unsigned int i=0;i<nR;i++)
	{
		if(Fixes[i]==1) FixEq.push_back(i);
		if(Fixes[i]==-1) FixVar.push_back( i>this->LeadStart?i+this->nLeader:i );
	}
	return this->LCPasMIP(FixEq, FixVar, solve);
}


GRBModel* LCP::LCPasMIP(
		vector<unsigned int> FixEq,	// If any equation is to be fixed to equality
		vector<unsigned int> FixVar, // If any variable is to be fixed to equality
		bool solve // Whether the model should be solved in the function already!
		)
/**
 * Uses the big M method to solve the complementarity problem. The variables and eqns to be set to equality can be given in FixVar and FixEq.
 * CAVEAT:
 * Note that the model returned by this function has to be explicitly deleted using the delete operator.
 */
{
	makeRelaxed();
	GRBModel *model = new GRBModel(this->RlxdModel);
	// Creating the model
	try{
		GRBVar x[nC], z[nR], u[nR];
		// Get hold of the Variables and Eqn Variables
		for(unsigned int i=0;i <nC;i++) x[i] = model->getVarByName("x_"+to_string(i));
		for(unsigned int i=0;i <nR;i++) z[i] = model->getVarByName("z_"+to_string(i));
		// Define binary variables for bigM
		for(unsigned int i=0;i <nR;i++) u[i] = model->addVar(0, 1, 1, GRB_BINARY, "u_"+to_string(i));
		// Include ALL Complementarity constraints using bigM
		for(auto p:Compl)
		{
			// z[i] <= Mu constraint
			GRBLinExpr expr = 0;
			expr += bigM*u[p.first];
			model->addConstr(expr, GRB_GREATER_EQUAL, z[p.first]);
			// x[i] <= M(1-u) constraint
			expr = bigM;
			expr -= bigM*u[p.first];
			model->addConstr(expr, GRB_GREATER_EQUAL, x[p.second]);
		}
		// If any equation or variable is to be fixed to zero, that happens here!
		for(auto i:FixVar) model->addConstr(x[i], GRB_EQUAL, 0.0);
		for(auto i:FixEq) model->addConstr(z[i], GRB_EQUAL, 0.0);
		model->update();
		if(solve) model->optimize();
	}
	catch(const char* e) { cout<<e<<endl; }
	catch(string e) { cout<<"String: "<<e<<endl; }
	catch(exception &e) { cout<<"Exception: "<<e.what()<<endl; }
	catch(GRBException &e){cout<<"GRBException: "<<e.getErrorCode()<<"; "<<e.getMessage()<<endl;}
	return model;
}

bool LCP::errorCheck(bool throwErr) const
{

	const unsigned int nR = M.n_rows;
	const unsigned int nC = M.n_cols;
	if(throwErr)
	{
		if(nR != q.n_rows) throw "M and q have unequal number of rows";
		if(nR + nLeader != nC) throw "Inconsistency between number of leader vars "+to_string(nLeader)+", number of rows "+to_string(nR)+" and number of cols "+to_string(nC);
	}
	return  (nR==q.n_rows && nR+nLeader == nC);
}


ostream& operator<<(ostream& ost, const LCP L)
{
	ost<<"LCP with "<<L.nR<<" rows and "<<L.nC<<" columns.";
	return ost;
}
