#include<iostream>
#include"func.h"
#include<armadillo>
#include<array>

using namespace std;

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
 * As per the convention, $y$ is the decision variable for the QP and 
 * that is parameterized in x
 * The KKT conditions arre
 * 0 <= y \perp  Mx + Ny + q >= 0
*/
{
	if (!this->dataCheck())
	{
		throw "Inconsistent data for KKT of QP_Param";
		return 0;
	}
	M = arma::join_cols(
			arma::join_rows(this->Q, -this->B),
			arma::join_rows(this->B.t(), arma::zeros<arma::sp_mat>(this->Ncons,this->Ncons))
		   );
	N = arma::join_rows(this->C, -this->A);
	q = arma::join_rows(this->c, this->b);
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





void MPEC(NashGame N, arma::sp_mat Q, QP_Param &P)
{
}

bool QP_Param::is_Playable(const QP_Param P) const
{
	unsigned int comp{static_cast<unsigned int>(P.getc().n_elem)};
	unsigned int compcomp {static_cast<unsigned int>(P.getB().n_cols)};
	if(this->Nx+ this->Ny == comp+ compcomp) // Total number of variables as we see as well as, as the competitor sees should be same.
		return(comp <= Ny && Nx <= compcomp); // Our size should at least be enemy's competition and vice versa.
	else return false;
}

int main22()
{
	QP_Param A;
	NashGame B(4);
	return 0;
}


unsigned int NashGame::FormulateLCP(arma::sp_mat &M, arma::vec &q) const
{
	vector<arma::sp_mat> Mi(Nplayers), Ni(Nplayers);
	vector<arma::vec> qi(Nplayers);
	vector<arma::sp_mat> tempM;
	vector<arma::vec> tempq;
	vector<unsigned int> PlayerSizes{};
	//
	// unsigned int Nvar = Players.at(0).getNx()+Players.at(0).getNy();
	// Get the KKT conditions for each player
	unsigned int NvarTot{0};
	PlayerSizes.push_back(NvarTot);
	for(unsigned int i=0; i<=Nplayers;i++)
	{
		this->Players[i]->KKT(Mi[i], Ni[i], qi[i]); 
		NvarTot += qi[i].n_elem;
		PlayerSizes.push_back(NvarTot);
	}
	M.set_size(NvarTot, NvarTot);
	q.set_size(NvarTot);
	for(unsigned int i=0; i<Nplayers;i++)
	{
		M.submat(PlayerSizes[i], PlayerSizes[i], PlayerSizes[i+1], PlayerSizes[i+1]) = Mi[i];
		q.rows(PlayerSizes[i], PlayerSizes[i+1]) = qi[i];
	} 
	return NvarTot;
}
