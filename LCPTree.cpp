#include<iostream>
#include<string>
#include"func.h"
#include<gurobi_c++.h>
#include<armadillo>

#undef VERBOSE
#define VERBOSE true

using namespace std;

bool operator == (vector<int> Fix1, vector<int> Fix2)
{
	if(Fix1.size() != Fix2.size()) return false;
	for(unsigned int i=0;i<Fix1.size();i++)
		if(Fix1[i]!=Fix2[i]) return false;
	return true;
}

bool operator < (vector<int> Fix1, vector<int> Fix2)
/**
 * Returns true if Fix1 is (grand) child of Fix2
 */
{
	if(Fix1.size() != Fix2.size()) return false;
	for(unsigned int i=0;i<Fix1.size();i++)
		if(Fix1[i]!=Fix2[i] && Fix2[i]!=0)
			return false; // Fix1 is not a child of Fix2
	return true;	 	// Fix1 is a child of Fix2
}

bool operator >(vector<int> Fix1, vector<int> Fix2)
{
	return (Fix2<Fix1);
}

vector<int>* LCP::anyBranch(const vector<vector<int>*>* vecOfFixes, vector<int>* Fix) const
/**
 * Returns true if any (grand)child of Fix is in vecOfFixes!
 */
{
	for(auto v:*vecOfFixes)
		if(*v < *Fix||*v==*Fix) return v;
	return NULL;
}

bool LCP::extractSols(GRBModel* model, arma::vec &z, arma::vec &x, bool extractZ) const
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

vector<int>* LCP::solEncode(const arma::vec &z, const arma::vec &x) const
{
	vector<signed int>* solEncoded = new vector<signed int>(nR, 0);
	for(auto p:Compl)
	{
		unsigned int i, j; i=p.first; j=p.second;
		if(isZero(z(i))) solEncoded->at(i)++;
		if(isZero(x(j))) solEncoded->at(i)--;
	}; 
	return solEncoded;
}

vector<int>* LCP::solEncode(GRBModel *model) const
{
	arma::vec x,z;
	if(!this->extractSols(model, z, x, true)) return {};// If infeasible model, return empty!
	else return this->solEncode(z,x);
}


void LCP::branch(int loc, const vector<int> *Fixes) 
/**
 * If loc == nR, then stop branching. We either hit infeasibility or a leaf.
 * If loc <0, then branch at abs(loc) location and go down the branch where variable is fixed to 0
 * else branch at abs(loc) location and go down the branch where eqn is fixed to 0
 */
{
	bool VarFirst=(loc<0);
	GRBModel *FixEqMdl=nullptr, *FixVarMdl=nullptr;
	vector<int> *FixEqLeaf, *FixVarLeaf;

	loc = (loc>=0)?loc:(loc==-(int)nR?0:-loc);
	if(VERBOSE) 
	{
		cout<<"Branching on Variable: "<<loc<<" with Fix as ";
		for(auto t1:*Fixes) cout<<t1<<"\t";
		cout<<endl;
	}
	if(loc >=(signed int)nR) return;
	GRBVar x,z;
	vector<int> *FixesEq = new vector<int>(*Fixes);
	vector<int> *FixesVar = new vector<int>(*Fixes);
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
}


vector<vector<int>*> *LCP::BranchAndPrune ()
{
	GRBModel *m = nullptr;
	vector<int>* Fix = new vector<int>(nR,0);
	branch(BranchLoc(m, Fix), Fix);
	return AllPolyhedra;
}

int LCP::BranchLoc(GRBModel* m, vector<int>* Fix)
{
	m = this->LCPasMIP(*Fix, true);
	int pos;
	pos = (signed int)nR;// Don't branch! You are at the leaf if pos never gets changed!!
	arma::vec z,x;
	if(this->extractSols(m, z, x, true)) // If already infeasible, nothing to branch!
	{
		vector<int> *v1 = this->solEncode(z,x);
		this->AllPolyhedra->push_back(v1);
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
	delete m;
	return pos; 
}


int LCP::BranchProcLoc(vector<int>* Fix, vector<int> *Leaf)
{
	int pos = (int)nR;
	if(*Fix==*Leaf) return nR;
	if(Fix->size()!=Leaf->size()) throw "Error in BranchProcLoc";
	for(unsigned int i=0;i<Fix->size();i++)
	{
		int l = Leaf->at(i);
		if(Fix->at(i)==0) return (l==0?-i:i*l);
	}
	return pos;
}
