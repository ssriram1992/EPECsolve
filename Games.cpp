#include<iostream>
#include"func.h"
#include<armadillo>
#include<array>

using namespace std;

template <class T> ostream& operator<<(ostream& ost, vector<T> v)
{
	for (auto elem:v) ost<<elem<<" ";
	ost<<endl;
	return ost;
}

template <class T, class S> ostream& operator<<(ostream& ost, pair<T,S> p)
{
	cout<<"<"<<p.first<<", "<<p.second<<">";
	return ost; 
}

ostream& operator<<(ostream& ost, perps C)
{
	 for (auto p:C)
		cout<<"<"<<p.first<<", "<<p.second<<">"<<"\t";
	return ost; 
}

ostream& operator<< (ostream& os, const QP_Param &Q)
{
	os<<"Quadratic program with linear inequality constraints: "<<endl;
	os<<Q.getNy()<<" decision variables parameterized by "<<Q.getNx()<<" variables"<<endl;
	os<<Q.getb().n_rows<<" linear inequalities"<<endl<<endl;
	if(VERBOSE)
	{
		Q.getQ().print("Q"); Q.getc().print("c"); Q.getC().print("C");
		Q.getA().print("A"); Q.getB().print("B"); Q.getb().print("b");
	}
	return os;
}

ostream& operator<< (ostream& os, const NashGame N)
{
	os<<endl;
	os<<"-----------------------------------------------------------------------"<<endl;
	os<<"Nash Game with "<<N.Nplayers<<" players"<<endl;
	os<<"-----------------------------------------------------------------------"<<endl;
	os<<"Number of primal variables:\t\t\t "<<N.primal_position.back()<<endl;
	os<<"Number of dual variables:\t\t\t "<<N.dual_position.back()-N.dual_position.front()+1<<endl;
	os<<"Number of shadow price dual variables:\t\t "<<N.MCRHS.n_rows<<endl;
	os<<"Number of leader variables:\t\t\t "<<N.n_LeadVar<<endl;
	os<<"-----------------------------------------------------------------------"<<endl;
	return os;
}


unsigned int QP_Param::size()
{
	this->Ny = this->Q.n_rows;
	this->Nx = this->C.n_cols;
	this->Ncons = this->b.size(); 
	return Ny;
}

bool QP_Param::dataCheck(bool forcesymm) const
{
	if(forcesymm && !this->Q.is_symmetric()) 
		return false; // Q should be symmetric if forcesymm is true
	if(this->A.n_cols != Nx) return false; 		// Rest are matrix size compatibility checks
	if(this->B.n_cols != Ny) return false;
	if(this->C.n_rows != Ny) return false;
	if(this->c.size() != Ny) return false;
	if(this->A.n_rows != Ncons) return false;
	if(this->B.n_rows != Ncons) return false;
	return true;
}

		
unsigned int QP_Param::KKT(arma::sp_mat& M, arma::sp_mat& N, arma::vec& q) const
/*
 * Writes the KKT condition of the parameterized QP
 * As per the convention, y is the decision variable for the QP and 
 * that is parameterized in x
 * The KKT conditions are
 * 0 <= y \perp  My + Nx + q >= 0
*/
{
	if (!this->dataCheck())
	{
		throw "Inconsistent data for KKT of QP_Param";
		return 0;
	}
	M = arma::join_cols( // In armadillo join_cols(A, B) is same as [A;B] in Matlab
			//  join_rows(A, B) is same as [A B] in Matlab
			arma::join_rows(this->Q, this->B.t()),
	  		arma::join_rows(-this->B, arma::zeros<arma::sp_mat>(this->Ncons, this->Ncons))
			);
	N = arma::join_cols(this->C, -this->A);
	q = arma::join_cols(this->c,  this->b);
	return M.n_rows;
}

arma::sp_mat QP_Param::getQ() const { return this->Q; } 
arma::sp_mat QP_Param::getC() const { return this->C; }
arma::sp_mat QP_Param::getA() const { return this->A; }
arma::sp_mat QP_Param::getB() const { return this->B; }
arma::vec QP_Param::getc() const { return this->c; }
arma::vec QP_Param::getb() const { return this->b; }
unsigned int QP_Param::getNx() const { return this->Nx; }
unsigned int QP_Param::getNy() const { return this->Ny; }

QP_Param& QP_Param::set(arma::sp_mat Q, arma::sp_mat C, arma::sp_mat A, arma::sp_mat B, arma::vec c, arma::vec b)
{
	this->Q = (Q); this->C = (C); this->A = (A);
	this->B = (B); this->c = (c); this->b = (b);
	this->size();
	if(!dataCheck()) throw "Bad initialization done in QP_Param::set";
	return *this;
}



QP_Param& QP_Param::setMove(arma::sp_mat Q, arma::sp_mat C, arma::sp_mat A, arma::sp_mat B, arma::vec c, arma::vec b)
{
	this->Q = move(Q); this->C = move(C); this->A = move(A);
	this->B = move(B); this->c = move(c); this->b = move(b);
	this->size();
	if(!dataCheck()) throw "Bad initialization done in QP_Param::set";
	return *this;
}



bool QP_Param::is_Playable(const QP_Param P) const
{
	unsigned int comp{static_cast<unsigned int>(P.getc().n_elem)};
	unsigned int compcomp {static_cast<unsigned int>(P.getB().n_cols)};
	if(this->Nx+ this->Ny == comp+ compcomp) // Total number of variables as we see as well as, as the competitor sees should be same.
		return(comp <= Ny && Nx <= compcomp); // Our size should at least be enemy's competition and vice versa.
	else return false;
}

NashGame::NashGame(vector<QP_Param*> Players, arma::sp_mat MC, arma::vec MCRHS, unsigned int n_LeadVar)
/*
 * Have a vector of QP_Param* ready such that
 * the variables are separated in x^{i} and x^{-i}
 * format.
 * In the correct ordering of variables, have the 
 * Market clearing equations ready. 
 * Now call this constructor.
 * It will allocate appropriate space for the dual variables 
 * for each player.
 * The ordering is 
 * [ Primal for Pl1, Primal for Pl2, ..., MarketCL duals,
 *  Dual for Pl1, Dual for Pl2, ... ]
 */
{
	// Setting the class variables
	this->n_LeadVar = n_LeadVar;
	this->Players = Players;
	this->Nplayers = Players.size();
	this->MarketClearing = MC;
	this->MCRHS = MCRHS;
	// Setting the size of class variable vectors
	this->primal_position.resize(this->Nplayers);
	this->dual_position.resize(this->Nplayers);
	// Defining the variable value
	unsigned int pr_cnt{0}, dl_cnt{0}; // Temporary variables - primal count and dual count
	vector<unsigned int> nCons(Nplayers); // Tracking the number of constraints in each player's problem
	for(unsigned int i=0; i<Nplayers;i++)
	{
		primal_position.at(i)=pr_cnt;
		pr_cnt += Players.at(i)->getNy();
		nCons.at(i) = Players.at(i)->getNx();
	}
	// Pushing back the end of primal position
	primal_position.push_back(pr_cnt);
	dl_cnt = pr_cnt; // From now on, the space is for dual variables.
	this->MC_dual_position = dl_cnt;
	this->Leader_position = dl_cnt+MCRHS.n_rows;
	dl_cnt += (MCRHS.n_rows + n_LeadVar);
	for(unsigned int i=0; i<Nplayers;i++)
	{
		dual_position.at(i) = dl_cnt;
		dl_cnt += Players.at(i)->getb().n_rows;
	}
	// Pushing back the end of dual position
	dual_position.push_back(dl_cnt);
}

unsigned int NashGame::FormulateLCP(arma::sp_mat &M, arma::vec &q, perps &Compl) const
{
	// To store the individual KKT conditions for each player.
	vector<arma::sp_mat> Mi(Nplayers), Ni(Nplayers); 
	vector<arma::vec> qi(Nplayers);
	
	unsigned int NvarFollow{0}, NvarLead{0};
	NvarLead = this->dual_position.back(); // Number of Leader variables (all variables)
	NvarFollow = NvarLead - this->n_LeadVar;
	M.set_size(NvarFollow, NvarLead);
	q.set_size(NvarFollow);
	M.zeros(); q.zeros(); // Make sure that the matrices don't have garbage value filled in !  
	// Get the KKT conditions for each player
	for(unsigned int i=0; i<Nplayers;i++)
	{
		this->Players[i]->KKT(Mi[i], Ni[i], qi[i]); 
		unsigned int Nprim, Ndual;
		Nprim = this->Players[i]->getNy();
		Ndual = this->Players[i]->getA().n_rows;
		// Adding the primal equations
		// Region 1 in Formulate LCP.ipe
		if(i>0) // For the first player, no need to add anything 'before' 0-th position
			M.submat(
				   	this->primal_position.at(i), 0,
					this->primal_position.at(i+1)-1, this->primal_position.at(i)-1
					) = Ni[i].submat(0,0,Nprim-1,this->primal_position.at(i)-1);
		// Region 2 in Formulate LCP.ipe
		M.submat(
				   	this->primal_position.at(i),  this->primal_position.at(i),
					this->primal_position.at(i+1)-1, this->primal_position.at(i+1)-1
				) = Mi[i].submat(0,0,Nprim-1,Nprim-1);
		// Region 3 in Formulate LCP.ipe
		M.submat(
					this->primal_position.at(i),  this->primal_position.at(i+1),
					this->primal_position.at(i+1)-1, this->dual_position.at(0)-1
				) = Ni[i].submat(0,this->primal_position.at(i),Nprim-1,Ni[i].n_cols-1);
		// Region 4 in Formulate LCP.ipe
		M.submat(
					this->primal_position.at(i),  this->dual_position.at(i),
					this->primal_position.at(i+1)-1, this->dual_position.at(i+1)-1
				) = Mi[i].submat(0, Nprim, Nprim-1, Nprim+Ndual-1);
		// RHS
		q.subvec(this->primal_position.at(i), this->primal_position.at(i+1)-1) = qi[i].subvec(0, Nprim-1);
		for(unsigned int j=this->primal_position.at(i);j<this->primal_position.at(i+1);j++)
			Compl.push_back({j, j});
		// Adding the dual equations
		// Region 5 in Formulate LCP.ipe
		if(i>0) // For the first player, no need to add anything 'before' 0-th position
			M.submat(
				   	this->dual_position.at(i)-n_LeadVar, 0,
					this->dual_position.at(i+1)-n_LeadVar-1, this->primal_position.at(i)-1
					) = Ni[i].submat(Nprim,0,Ni[i].n_rows-1,this->primal_position.at(i)-1);
		// Region 6 in Formulate LCP.ipe
		M.submat(
				   	this->dual_position.at(i)-n_LeadVar,  this->primal_position.at(i),
					this->dual_position.at(i+1)-n_LeadVar-1, this->primal_position.at(i+1)-1
				) = Mi[i].submat(Nprim,0,Nprim+Ndual-1,Nprim-1);
		// Region 7 in Formulate LCP.ipe
		M.submat(
					this->dual_position.at(i)-n_LeadVar,  this->primal_position.at(i+1),
					this->dual_position.at(i+1)-n_LeadVar-1, this->dual_position.at(0)-1
				) = Ni[i].submat(Nprim,this->primal_position.at(i),Ni[i].n_rows-1,Ni[i].n_cols-1);
		// Region 8 in Formulate LCP.ipe
		M.submat(
					this->dual_position.at(i)-n_LeadVar,  this->dual_position.at(i),
					this->dual_position.at(i+1)-n_LeadVar-1, this->dual_position.at(i+1)-1
				) = Mi[i].submat(Nprim, Nprim, Nprim+Ndual-1, Nprim+Ndual-1);
		// RHS
		q.subvec(this->dual_position.at(i)-n_LeadVar, this->dual_position.at(i+1)-n_LeadVar-1) = qi[i].subvec(Nprim, qi[i].n_rows-1);
		for(unsigned int j=this->dual_position.at(i)-n_LeadVar;j<this->dual_position.at(i+1)-n_LeadVar;j++)
			Compl.push_back({j, j+n_LeadVar});
	}
	M.submat(this->MC_dual_position,0,this->Leader_position-1,this->dual_position.at(0)-1) = this->MarketClearing;
	q.subvec(this->MC_dual_position,this->Leader_position-1) = -this->MCRHS;
	for(unsigned int j=this->MC_dual_position;j<this->Leader_position;j++)
		Compl.push_back({j, j}); 
	return NvarFollow;
}
