#include<iostream>
#include<memory>
#include<string>
#include"func.h"
#include<gurobi_c++.h>
#include<armadillo>

using namespace std;


void 
LCP::defConst(GRBEnv* env)
{
	AllPolyhedra = new vector<vector<short int>*> {};
	RelAllPol = new vector<vector<short int>*> {};
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

LCP::LCP(GRBEnv *env, NashGame N):RlxdModel(*env)
{
	arma::sp_mat M; arma::vec q; perps Compl;
	N.FormulateLCP(M, q, Compl);
}

/** @brief Destructor of LCP */
/** LCP object owns the pointers to definitions of its polyhedra that it owns
 It has to be deleted and freed. */
LCP::~LCP()
{
	for(auto p:*(this->AllPolyhedra)) delete p;
	for(auto p:*(this->RelAllPol)) delete p;
	delete this->AllPolyhedra;
	delete this->RelAllPol;
	for(auto a:*(this->Ai)) delete a; 
	for(auto b:*(this->bi)) delete b;
	delete Ai; delete bi;
}

/** @brief Makes a Gurobi object that relaxes complementarity constraints in an LCP */
/** @details A Gurobi object is stored in the LCP object, that has all complementarity constraints removed.
 * A copy of this object is used by other member functions */
int 
LCP::makeRelaxed()
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
		// If @f$Ax \leq b@f$ constraints are there, they should be included too!
		if(this->_A.n_nonzero != 0 || this->_b.n_rows!=0)
		{ 
			if(_A.n_cols != nC || _A.n_rows != _b.n_rows) throw "A and b are incompatible! Thrown from makeRelaxed()";
			for(unsigned int i=0;i<_A.n_rows;i++)
			{
				GRBLinExpr expr = 0;
				for(auto a=_A.begin_row(i); a!=_A.end_row(i); ++a)
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
	if(!this->madeRlxdModel) throw "Error in LCP::makeRelaxed";
	return 0;
}

/** 
 * The returned model has constraints
 * corresponding to the indices in FixEq set to equality
 * and variables corresponding to the indices
 * present in FixVar set to equality (=0)
 */
/// @warning The FixEq and FixVar variables are used under a different convention here!
/// @warning This member function is public for the moment. But this will be converted to a private method soon.
unique_ptr<GRBModel> 
LCP::LCP_Polyhed_fixed(
        /// If index is present, equality imposed on that variable
		vector<unsigned int> FixEq,  		
        /// If index is present, equality imposed on that equation
		vector<unsigned int> FixVar  		
		)			
{
	makeRelaxed();
	unique_ptr<GRBModel> model(new GRBModel(this->RlxdModel));
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

/**
 * Returs a model created from a given model
 * The returned model has constraints
 * corresponding to the non-zero elements of FixEq set to equality
 * and variables corresponding to the non-zero
 * elements of FixVar set to equality (=0)
 */
unique_ptr<GRBModel> 
LCP::LCP_Polyhed_fixed(
        /// If non zero, equality imposed on variable
		arma::Col<int> FixEq,  		
        /// If non zero, equality imposed on equation
		arma::Col<int> FixVar  		
		)			
{
	makeRelaxed();
	unique_ptr<GRBModel> model{new GRBModel(this->RlxdModel)};
	for(unsigned int i=0;i<nC;i++)
		if(FixVar[i]) 
			model->getVarByName("x_"+to_string(i)).set(GRB_DoubleAttr_UB,0);
	for(unsigned int i=0;i<nR;i++)
		if(FixEq[i]) 
			model->getVarByName("z_"+to_string(i)).set(GRB_DoubleAttr_UB,0);
	model->update();
	return model;
}

/**
 * Uses the big M method to solve the complementarity problem. The variables and eqns to be set to equality can be given in Fixes in 0/+1/-1 notation
 * @warning Note that the model returned by this function has to be explicitly deleted using the delete operator.
 */
unique_ptr<GRBModel> 
LCP::LCPasMIP(vector<short int> Fixes, bool solve)
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


/**
 * Uses the big M method to solve the complementarity problem. The variables and eqns to be set to equality can be given in FixVar and FixEq.
 * @warning Note that the model returned by this function has to be explicitly deleted using the delete operator.
 */
unique_ptr<GRBModel> 
LCP::LCPasMIP(
		vector<unsigned int> FixEq,	// If any equation is to be fixed to equality
		vector<unsigned int> FixVar, // If any variable is to be fixed to equality
		bool solve // Whether the model should be solved in the function already!
		)
{
	makeRelaxed();
	unique_ptr<GRBModel> model{new GRBModel(this->RlxdModel)};
	// Creating the model
	try
	{
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
		return model;
	}
	catch(const char* e) { cout<<e<<endl; }
	catch(string e) { cout<<"String: "<<e<<endl; }
	catch(exception &e) { cout<<"Exception: "<<e.what()<<endl; }
	catch(GRBException &e){cout<<"GRBException: "<<e.getErrorCode()<<"; "<<e.getMessage()<<endl;}
	throw "Error in LCP::LCPasMIP";
	return nullptr;
}

/**
 * Checks if the `M` and `q` given to create the LCP object are of 
 * compatible size, given the number of leader variables
 */
bool 
LCP::errorCheck(
        /// If this is true, function throws an error, else, it just returns false
        bool throwErr) const
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


void 
LCP::print(string end)
{
	cout<<"LCP with "<<this->nR<<" rows and "<<this->nC<<" columns."<<end;
}

/** @warning Computes convex hull of LCP feasible region */
int 
ConvexHull(
		vector<arma::sp_mat*> *Ai, vector<arma::vec*> *bi, // Individual constraints
		arma::sp_mat &A, arma::vec &b, // To store outputs
		arma::sp_mat Acom, arma::vec bcom // Common constraints.
		)
{
	// Count number of polyhedra and the space we are in!
	unsigned int nPoly{static_cast<unsigned int>(Ai->size())};
	unsigned int nC{static_cast<unsigned int>(Ai->front()->n_cols)};

	// Count the number of variables in the convex hull.
	unsigned int nFinCons{0}, nFinVar{0};
	// Error check
	if (nPoly == 0) 					throw "Empty vector of polyhedra given!";	// There should be at least 1 polyhedron to consider
	if (nPoly != bi->size()) 			throw "Inconsistent number of LHS and RHS for polyhedra"; 
	for(unsigned int i=0; i!=nPoly; i++)
	{
		if (Ai->at(i)->n_cols != nC) 		throw "Inconsistent number of variables in the polyhedra " + to_string(i) + "; " + to_string(Ai->at(i)->n_cols)+"!="+to_string(nC);
		if (Ai->at(i)->n_rows != bi->at(i)->n_rows) throw "Inconsistent number of rows in LHS and RHS of polyhedra " + to_string(i)+";" + to_string(Ai->at(i)->n_rows) + "!=" + to_string(bi->at(i)->n_rows);
		nFinCons += Ai->at(i)->n_rows;
	} 	
	unsigned int FirstCons = nFinCons;
	if(Acom.n_rows >0 &&Acom.n_cols != nC) throw "Inconsistent number of variables in the common polyhedron";
	if(Acom.n_rows >0 &&Acom.n_rows != bcom.n_rows) throw "Inconsistent number of rows in LHS and RHS in the common polyhedron";

	// 2nd constraint in Eqn 4.31 of Conforti - twice so we have 2 ineq instead of 1 eq constr
	nFinCons += nC*2;
	// 3rd constr in Eqn 4.31. Again as two ineq constr.
	nFinCons += 2;
	// Common constraints
	nFinCons += Acom.n_rows;

	nFinVar = nPoly*nC + nPoly + nC; // All x^i variables + delta variables+ original x variables 
	A.resize(nFinCons, nFinVar); b.resize(nFinCons);
	A.zeros(); b.zeros();
	// Counting rows completed
	unsigned int complRow{0};
	if (VERBOSE){cout<<"In Convex Hull computation!"<<endl;}
	for(unsigned int i = 0; i<nPoly; i++)
	{
		unsigned int nConsInPoly = complRow+Ai->at(i)->n_rows ;
		// First constraint in (4.31)
		A.submat(complRow, i*nC, complRow+nConsInPoly-1, (i+1)*nC-1) = *Ai->at(i);
		// First constraint RHS
		A.submat(complRow, nPoly*nC+i, complRow+nConsInPoly-1, nPoly*nC+i) = -*bi->at(i);
		// Second constraint in (4.31)
		for(unsigned int j=0; j<nC;j++)
		{
			A.at(FirstCons+2*j,(i*nC)+j) = 1;
			A.at(FirstCons+2*j+1,(i*nC)+j) = -1;
		}
		// Third constraint in (4.31)
		A.at(FirstCons + nC*2, nPoly*nC + i) = 1;
		A.at(FirstCons + nC*2 + 1, nPoly*nC + i) = -1;
	}
	// Second Constraint RHS
	for(unsigned int j=0; j<nC;j++)
		A.at(FirstCons+2*j, nPoly*nC + nPoly +j) = -1;
	// Third Constraint RHS
	b.at(FirstCons + nC*2) = 1;
	b.at(FirstCons + nC*2+1) = -1;
	// Common Constraints
	if(Acom.n_rows>0)
	{
		b.subvec(FirstCons+2*nC+2, nFinCons-1) = bcom;
		A.submat(FirstCons+2*nC+2, nPoly*nC+nPoly,
					nFinCons-1, nFinVar-1) = Acom;
	}
	return 0;
}

arma::vec 
isFeas(const arma::sp_mat* A, const arma::vec *b, const arma::vec *c, bool Positivity)
{
	unsigned int nR, nC;
	nR = A->n_rows; nC = A->n_cols;
	if(c->n_rows != nC) throw "Inconsistency in no of Vars in isFeas()";
	if(b->n_rows != nR) throw "Inconsistency in no of Constr in isFeas()";

	arma::vec sol = arma::vec(c->n_rows, arma::fill::zeros);
	const double lb = Positivity?0:-GRB_INFINITY;

	GRBEnv env;
	GRBModel model = GRBModel(env);
	GRBVar x[nC];
	GRBConstr a[nR];
	// Adding Variables
	for(unsigned int i=0; i<nC; i++)
		x[i] = model.addVar(lb, GRB_INFINITY, c->at(i), GRB_CONTINUOUS, "x_"+to_string(i));
	// Adding constraints
	for(unsigned int i=0; i<nR; i++)
	{
		GRBLinExpr lin{0};
		for(auto j=A->begin_row(i); j!=A->end_row(i);++j)
			lin += (*j)*x[j.col()];
		a[i] = model.addConstr(lin, GRB_LESS_EQUAL, b->at(i));
	}
	model.set(GRB_IntParam_OutputFlag, 0 ) ;
	model.set(GRB_IntParam_DualReductions, 0) ;
	model.optimize();
	if(model.get(GRB_IntAttr_Status)==GRB_OPTIMAL)
		for(unsigned int i=0; i<nC; i++) sol.at(i) = x[i].get(GRB_DoubleAttr_X); 
	return sol;
}


bool 
operator == (vector<int> Fix1, vector<int> Fix2)
{
	if(Fix1.size() != Fix2.size()) return false;
	for(unsigned int i=0;i<Fix1.size();i++)
		if(Fix1[i]!=Fix2[i]) return false;
	return true;
}

/**
 * Returns true if Fix1 is (grand) child of Fix2
 *  Defn Grand Parent:
 *  	Either the same value as the grand child, or has 0 in that location
 *  Defn Grand child:
 *  	Same val as grand parent in every location, except any val allowed, if grandparent is 0
 */
bool 
operator < (vector<int> Fix1, vector<int> Fix2)
{
	if(Fix1.size() != Fix2.size()) return false;
	for(unsigned int i=0;i<Fix1.size();i++)
		if(Fix1[i]!=Fix2[i] && Fix1[i]*Fix2[i]!=0)
			return false; // Fix1 is not a child of Fix2
	return true;	 	// Fix1 is a child of Fix2
}

bool 
operator >(vector<int> Fix1, vector<int> Fix2)
{
	return (Fix2<Fix1);
}

/** @brief Returns true if any (grand)child of Fix is in vecOfFixes!  */
/**
 *  Defn Grand Parent:
 *  	Either the same value as the grand child, or has 0 in that location
 *
 *  Defn Grand child:
 *  	Same val as grand parent in every location, except any val allowed, if grandparent is 0
 */
vector<short int>* 
LCP::anyBranch(const vector<vector<short int>*>* vecOfFixes, vector<short int>* Fix) const
{
	for(auto v:*vecOfFixes)
		if(*Fix < *v||*v==*Fix) return v;
	return NULL;
}

/** @brief Extracts variable and equation values from a solved Gurobi model for LCP */
bool 
LCP::extractSols(
        /// The Gurobi Model that was solved (perhaps using LCP::LCPasMIP)
        GRBModel* model, 
        /// Output variable - where the equation values are stored
        arma::vec &z, 
        /// Output variable - where the variable values are stored
        arma::vec &x, 
        /// z values are filled only if this is true
        bool extractZ) const
{
	if(model->get(GRB_IntAttr_Status) == GRB_LOADED) model->optimize();
	if(model->get(GRB_IntAttr_Status) != GRB_OPTIMAL) return false;
	x.set_size(nC); if(extractZ) z.set_size(nR);
	for(unsigned int i=0; i<nR;i++)
	{
		x[i] = model->getVarByName("x_"+to_string(i)).get(GRB_DoubleAttr_X);
		if(extractZ) z[i] = model->getVarByName("z_"+to_string(i)).get(GRB_DoubleAttr_X);
	}
	for(unsigned int i=nR;i<nC;i++)
		x[i] = model->getVarByName("x_"+to_string(i)).get(GRB_DoubleAttr_X); 
	return true;
}

/// @brief Given variable values and equation values, encodes it in 0/+1/-1 format and returns it.
vector<short int>* 
LCP::solEncode(const arma::vec &z, const arma::vec &x) const
{
	vector<signed short int>* solEncoded = new vector<signed short int>(nR, 0);
	for(auto p:Compl)
	{
		unsigned int i, j; i=p.first; j=p.second;
		if(isZero(z(i))) solEncoded->at(i)++;
		if(isZero(x(j))) solEncoded->at(i)--;
	}; 
	return solEncoded;
}

/// @brief Given a Gurobi model, extracts variable values and equation values, encodes it in 0/+1/-1 format and returns it.
vector<short int>* 
LCP::solEncode(GRBModel *model) const
{
	arma::vec x,z;
	if(!this->extractSols(model, z, x, true)) return {};// If infeasible model, return empty!
	else return this->solEncode(z,x);
}

/** @internal
 * If loc == nR, then stop branching. We either hit infeasibility or a leaf.
 * If loc <0, then branch at abs(loc) location and go down the branch where variable is fixed to 0
 * else branch at abs(loc) location and go down the branch where eqn is fixed to 0
 */
void 
LCP::branch(int loc, const vector<short int> *Fixes) 
{
	bool VarFirst=(loc<0);
	unique_ptr<GRBModel> FixEqMdl, FixVarMdl;
	// GRBModel *FixEqMdl=nullptr, *FixVarMdl=nullptr;
	vector<short int> *FixEqLeaf, *FixVarLeaf;
	if(VERBOSE) 
	{
		cout<<endl<<"Branching on Variable: "<<loc<<" with Fix as ";
		for(auto t1:*Fixes) cout<<t1<<"\t";
		cout<<endl;
	}

	loc = (loc>=0)?loc:(loc==-(int)nR?0:-loc);
	if(loc >=(signed int)nR) 
	{
		if(VERBOSE)
			cout<<"nR: "<<nR<<"\tloc: "<<loc<<"\t Returning..."<<endl; 
		return;
	}
	else
	{
	GRBVar x,z;
	vector<short int> *FixesEq = new vector<short int>(*Fixes);
	vector<short int> *FixesVar = new vector<short int>(*Fixes);
	if(Fixes->at(loc) != 0) throw "Fixing an already fixed variable!";
	FixesEq->at(loc) =1; FixesVar->at(loc)=-1;
	if(VarFirst)
	{
		FixVarLeaf = anyBranch(AllPolyhedra, FixesVar);
		if(!FixVarLeaf) this->branch(BranchLoc(FixVarMdl, FixesVar), FixesVar); 
		else this->branch(BranchProcLoc(FixesVar, FixVarLeaf),FixesVar);

		FixEqLeaf = anyBranch(AllPolyhedra, FixesEq);
		if(!FixEqLeaf) this->branch(BranchLoc(FixEqMdl, FixesEq), FixesEq);
		else this->branch(BranchProcLoc(FixesEq, FixEqLeaf),FixesEq);
	}
	else
	{
		FixEqLeaf = anyBranch(AllPolyhedra, FixesEq);
		if(!FixEqLeaf) this->branch(BranchLoc(FixEqMdl, FixesEq), FixesEq);
		else this->branch(BranchProcLoc(FixesEq, FixEqLeaf),FixesEq);

		FixVarLeaf = anyBranch(AllPolyhedra, FixesVar);
		if(!FixVarLeaf) this->branch(BranchLoc(FixVarMdl, FixesVar), FixesVar); 
		else this->branch(BranchProcLoc(FixesVar, FixVarLeaf),FixesVar); 
	}
	delete FixesEq;
	delete FixesVar;
	}
}

vector<vector<short int>*>* 
LCP::BranchAndPrune ()
{
	unique_ptr<GRBModel> m;
	vector<short int>* Fix = new vector<short int>(nR,0);
	branch(BranchLoc(m, Fix), Fix);
	delete Fix;
	return AllPolyhedra;
}

int 
LCP::BranchLoc(unique_ptr<GRBModel> &m, vector<short int>* Fix)
{
	static int GurCallCt {0};
	m = this->LCPasMIP(*Fix, true);
	GurCallCt++;
	if(VERBOSE)
	{
		cout<<"Gurobi call\t"<<GurCallCt<<"\t";
		for (auto a:*Fix) cout<<a<<"\t";
		cout<<endl;
	}
	int pos;
	pos = (signed int)nR;// Don't branch! You are at the leaf if pos never gets changed!!
	arma::vec z,x;
	if(this->extractSols(m.get(), z, x, true)) // If already infeasible, nothing to branch!
	{
		vector<short int> *v1 = this->solEncode(z,x);
		vector<short int> *v2 = anyBranch(AllPolyhedra, v1);
		if(VERBOSE)
		{
			cout<<"v1: \t\t\t";
			for (auto a:*v1) cout<<a<<"\t";
			cout<<"\t\t";
			cout<<"v2: \t\t\t";
			if(v2) for (auto a:*v2) cout<<a<<"\t";
			else cout<<"NULL";
			cout<<endl;
		}
		// if(v2==NULL)
		{
			this->AllPolyhedra->push_back(v1);
			this->FixToPolies(v1);
			
			if(VERBOSE)
			{
				cout<<"New Polyhedron found"<<endl;
				x.t().print("x");z.t().print("z");
			}
		}
		////////////////////
		// BRANCHING RULE //
		////////////////////
		
		// Branch at a large positive value
		double maxvalx{0}; unsigned int maxposx{nR};
		double maxvalz{0}; unsigned int maxposz{nR};
		for (unsigned int i=0;i<nR;i++)
		{
			unsigned int varPos = i>=this->LeadStart?i+nLeader:i;
			if(x(varPos) > maxvalx && Fix->at(i)==0) // If already fixed, it makes no sense!
			{
				maxvalx = x(varPos); 
				maxposx = (i==0)?-nR:-i; // Negative of 0 is -nR by convention
			}
			if(z(i) > maxvalz && Fix->at(i)==0) // If already fixed, it makes no sense!
			{
				maxvalz = z(i); 
				maxposz = i;
			}
		}
		pos = maxvalz>maxvalx? maxposz:maxposx;
		///////////////////////////
		// END OF BRANCHING RULE //
		///////////////////////////
	}
	else 
	{
		if(VERBOSE)
			cout<<"Infeasible branch"<<endl;
	}
	// delete m; // Since we moved to unique_ptr
	return pos; 
}

int 
LCP::BranchProcLoc(vector<short int>* Fix, vector<short int> *Leaf)
{
	int pos = (int)nR;
	if(VERBOSE)
	{
		cout<<"Processed Node \t\t";
		for(auto a:*Fix) cout<<a<<"\t";
		cout<<endl;
	}
	if(*Fix==*Leaf) return nR;
	if(Fix->size()!=Leaf->size()) throw "Error in BranchProcLoc";
	for(unsigned int i=0;i<Fix->size();i++)
	{
		int l = Leaf->at(i);
		if(Fix->at(i)==0) return (l==0?-i:i*l);
	}
	return pos;
}

void 
LCP::FixToPoly(const vector<short int> *Fix, bool checkFeas, bool custom, vector<arma::sp_mat*> *custAi, vector<arma::vec*> *custbi)
{
	arma::sp_mat *Aii = new arma::sp_mat(nR, nC);
   	arma::vec *bii = new arma::vec(nR, arma::fill::zeros);
	for(unsigned int i=0;i<this->nR;i++)
	{
		if(Fix->at(i) == 0) throw "Error in FixToPoly";
		if(Fix->at(i)==1) // Equation to be fixed top zero
		{
			for(auto j=this->M.begin_row(i); j!=this->M.end_row(i); ++j)
				if(!this->isZero((*j))) Aii->at(i, j.col()) = (*j); // Only mess with non-zero elements of a sparse matrix!
			bii->at(i) = this->q(i);
		}
		else // Variable to be fixed to zero, i.e. x(j) <= 0 constraint to be added
		{
			unsigned int varpos = (i>this->LeadStart)?i+this->nLeader:i;
			Aii->at(i, varpos) = 1; 
			bii->at(i) = 0;
		}
	}
	bool add = !checkFeas;
	if(checkFeas)
	{
		bool Error{true};
		unsigned int count{0};
		try
		{
			makeRelaxed();
			GRBModel* model = new GRBModel(this->RlxdModel);
			for(auto i:*Fix)
			{
				if(i>0) // Fixing the eqn to zero
					model->getVarByName("z_"+to_string(count)).set(GRB_DoubleAttr_UB,0);
				if(i<0)
					model->getVarByName("x_"+to_string(count>this->LeadStart?count+nLeader:i)).set(GRB_DoubleAttr_UB,0);
				count++;
			} 
			model->optimize();
			if(model->get(GRB_IntAttr_Status) == GRB_OPTIMAL) add = true;
			delete model;
			Error = false;
		}
		catch(const char* e) { cout<<e<<endl; }
		catch(string e) { cout<<"String: "<<e<<endl; }
		catch(exception &e) { cout<<"Exception: "<<e.what()<<endl; }
		catch(GRBException &e) {cout<<"GRBException: "<<e.getErrorCode()<<": "<<e.getMessage()<<endl;}
		if(Error) throw "Error in LCP::FixToPoly";
	}
	if(add) 
	{ 
		custom?this->Ai->push_back(Aii):custAi->push_back(Aii); 
		custom?this->bi->push_back(bii):custbi->push_back(bii); 
	}
	if(VERBOSE) cout<<"Pushed a new polyhedron! No: "<<Ai->size()<<endl;
}

void 
LCP::FixToPolies(const vector<short int> *Fix, bool checkFeas, bool custom, vector<arma::sp_mat*> *custAi, vector<arma::vec*> *custbi)
{
	bool flag = false;
	vector<short int> MyFix(*Fix);
	unsigned int i;
	for(i=0; i<this->nR;i++)
		if(Fix->at(i)==0) { flag = true; break; }
	if(flag)
	{
		MyFix[i] = 1;
		this->FixToPolies(&MyFix, checkFeas, custom, custAi, custbi);
		MyFix[i] = -1;
		this->FixToPolies(&MyFix, checkFeas, custom, custAi, custbi);
	}
	else this->FixToPoly(Fix, checkFeas, custom, custAi, custbi);
}

int 
LCP::EnumerateAll(const bool solveLP)
{
	delete Ai; delete bi; // Just in case it is polluted with BranchPrune
	Ai = new vector<arma::sp_mat *>{}; bi = new vector<arma::vec *>{};
	vector<short int> *Fix = new vector<short int>(nR,0);
	this->FixToPolies(Fix, solveLP);
	return 0;
}

void LCP::addPolyhedron(const vector<short int> &Fix, vector<arma::sp_mat*> &custAi, vector<arma::vec*> &custbi,
				const bool convHull, arma::sp_mat *A, arma::vec  *b)
{
	this->FixToPolies(&Fix, false, true, &custAi, &custbi);
	if(convHull)
		::ConvexHull(&custAi, &custbi, *A, *b, this->_A, this->_b);
}

unique_ptr<GRBModel> 
LCP::LCPasQP(bool solve)
/** @brief Solves the LCP as a QP using Gurobi */
/** Removes all complementarity constraints from the QP's constraints. Instead, the sum of products of complementarity pairs is minimized. If the optimal value turns out to be 0, then it is actually a solution of the LCP. Else the LCP is infeasible.  
 * @warning Solves the LCP feasibility problem. Not the MPEC optimization problem.
 * */
{
	this->makeRelaxed();
	unique_ptr<GRBModel> model(new GRBModel(this->RlxdModel));
	GRBQuadExpr obj = 0;
	GRBVar x[this->nR];
	GRBVar z[this->nR];
	for(auto p:this->Compl)
	{
		unsigned int i=p.first; unsigned int j = p.second;
		z[i] = model->getVarByName("z_"+to_string(i));
		x[i] = model->getVarByName("x_"+to_string(j));
		obj += x[i]*z[i];
	}
	model->setObjective(obj, GRB_MINIMIZE);
	bool Error{false};
	if(solve) 
	{
		Error = true;
		try
		{
			model->optimize();
			int status = model->get(GRB_IntAttr_Status);
			if(status!=GRB_OPTIMAL || model->get(GRB_DoubleAttr_ObjVal) > this->eps)
				throw "LCP infeasible";
			Error = false;
		}
		catch(const char* e) { cout<<e<<endl; }
		catch(string e) { cout<<"String: "<<e<<endl; }
		catch(exception &e) { cout<<"Exception: "<<e.what()<<endl; }
		catch(GRBException &e){cout<<"GRBException: "<<e.getErrorCode()<<"; "<<e.getMessage()<<endl;}
		if(Error) throw "Error in LCP::LCPasQP";
	}
	return model;
}

unique_ptr<GRBModel>
LCP::LCPasMIP(bool solve)
{
	return this->LCPasMIP({}, {}, solve);
}


unique_ptr<GRBModel> 
LCP::MPECasMILP(const arma::sp_mat &C, const arma::vec &c, const arma::vec &x_minus_i, bool solve)
{
	unique_ptr<GRBModel> model = this->LCPasMIP(false);
	arma::vec Cx(this->nC, arma::fill::zeros);
	try 
	{
		Cx = C*x_minus_i; 
		if(Cx.n_rows != this->nC) throw string("Bad size of C");
		if(c.n_rows != this->nC) throw string("Bad size of c");
	}
	catch(exception &e) {cout<<"Exception in LCP::MPECasMIQP: "<<e.what()<<endl;throw;}
	catch(string &e) {cout<<"Exception in LCP::MPECasMIQP: "<<e<<endl;throw;}
	arma::vec obj = c+Cx;
	GRBLinExpr expr{0};
	for(unsigned int i=0; i<this->nC;i++)
		expr += obj.at(i)*model->getVarByName("x_"+to_string(i));
	model->setObjective(expr, GRB_MINIMIZE);
	if(solve) model->optimize();
	return model;
}

unique_ptr<GRBModel> 
LCP::MPECasMIQP(const arma::sp_mat &Q, const arma::sp_mat &C, const arma::vec &c, const arma::vec &x_minus_i, bool solve)
{
	auto model = this->MPECasMILP(C, c, x_minus_i, false);
	/// Note that if the matrix Q is a zero matrix, then this returns a Gurobi MILP model as opposed to MIQP model.
	/// This enables Gurobi to use its much advanced MIP solver
	if(Q.n_nonzero != 0) // If Q is zero, then just solve MIP as opposed to MIQP!
	{
		GRBLinExpr linexpr = model->getObjective(0);
		GRBQuadExpr expr{linexpr};
		for(auto it = Q.begin(); it!=Q.end(); ++it)
			expr += (*it)*
					model->getVarByName("x_" + to_string(it.row()))*
					model->getVarByName("x_" + to_string(it.col()));
		model->setObjective(expr, GRB_MINIMIZE);
	}
	if(solve) model->optimize();
	return model;
} 
