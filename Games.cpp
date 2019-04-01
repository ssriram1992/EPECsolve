#include<iostream>
#include<memory>
#include"func.h"
#include<armadillo>
#include<array>

using namespace std;

template 
<class T> 
ostream& 
operator<<(ostream& ost, vector<T> v)
{
	for (auto elem:v) ost<<elem<<" ";
	ost<<endl;
	return ost;
}

template 
<class T, class S> 
ostream& 
operator<<(ostream& ost, pair<T,S> p)
{
	cout<<"<"<<p.first<<", "<<p.second<<">";
	return ost; 
}

ostream& 
operator<<(ostream& ost, perps C)
{
	 for (auto p:C)
		cout<<"<"<<p.first<<", "<<p.second<<">"<<"\t";
	return ost; 
}

ostream& 
operator<< (ostream& os, const Game::QP_Param &Q)
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


Game::MP_Param& Game::MP_Param::addDummy(unsigned int pars, unsigned int vars)
/**
 * Adds dummy variables to a parameterized mathematical program
 */
{
	this->Nx += pars;
	this->Ny += vars;
	if(vars)
	{
		Q.resize(this->Ny, this->Ny);
		B.resize(this->Ncons, this->Ny);
		c.resize(this->Ny);
	}
	if(pars)
		A.resize(this->Ncons, this->Nx);
	if(vars || pars)
		C.resize(this->Ny, this->Nx); 

	return *this;
}


unsigned int 
Game::MP_Param::size()
/** @brief Calculates @p Nx, @p Ny and @p Ncons
 *	Computes parameters in MP_Param:
 *		- Computes @p Ny as number of rows in MP_Param::Q
 * 		- Computes @p Nx as number of columns in MP_Param::C
 * 		- Computes @p Ncons as number of rows in MP_Param::b, i.e., the RHS of the constraints
 *
 * 	For proper working, MP_Param::dataCheck() has to be run after this.
 * 	@returns @p Ny, Number of variables in the quadratic program, QP
 */
{
	this->Ny = this->Q.n_rows;
	this->Nx = this->C.n_cols;
	this->Ncons = this->b.size(); 
	return Ny;
}


Game::MP_Param& 
Game::MP_Param::set(arma::sp_mat &Q, arma::sp_mat &C, arma::sp_mat &A, arma::sp_mat &B, arma::vec &c, arma::vec &b)
/// Setting the data, while keeping the input objects intact
{
	this->Q = (Q); this->C = (C); this->A = (A);
	this->B = (B); this->c = (c); this->b = (b);
	if(!finalize()) throw string("Error in MP_Param::set: Invalid data");
	return *this;
}



Game::MP_Param& 
Game::MP_Param::set(arma::sp_mat &&Q, arma::sp_mat &&C, arma::sp_mat &&A, arma::sp_mat &&B, arma::vec &&c, arma::vec &&b)
/// Faster means to set data. But the input objects might be corrupted now.
{
	this->Q = move(Q); this->C = move(C); this->A = move(A);
	this->B = move(B); this->c = move(c); this->b = move(b);
	if(!finalize()) throw string("Error in MP_Param::set: Invalid data");
	return *this;
}


bool 
Game::MP_Param::dataCheck(bool forcesymm ///< Check if MP_Param::Q is symmetric
		) const
/** @brief Check that the data for the MP_Param class is valid
 * Always works after calls to MP_Param::size()
 * Checks that are done:
 * 		- Number of columns in @p Q is same as @p Ny (Q should be square)
 * 		- Number of columns of @p A should be @p Nx
 * 		- Number of columns of @p B should be @p Ny
 * 		- Number of rows in @p C should be @p Ny
 * 		- Size of @p c should be @p Ny
 * 		- @p A and @p B should have the same number of rows, equal to @p Ncons
 * 		- if @p forcesymm is @p true, then Q should be symmetric
 *
 * 	@returns true if all above checks are cleared. false otherwise.
 */
{
	if(forcesymm && !this->Q.is_symmetric()) 
		return false; // Q should be symmetric if forcesymm is true
	if(this->Q.n_cols != Ny) return false;
	if(this->A.n_cols != Nx) return false; 		// Rest are matrix size compatibility checks
	if(this->B.n_cols != Ny) return false;
	if(this->C.n_rows != Ny) return false;
	if(this->c.size() != Ny) return false;
	if(this->A.n_rows != Ncons) return false;
	if(this->B.n_rows != Ncons) return false;
	return true;
}

int
Game::QP_Param::make_yQy()
	/// Adds the Gurobi Quadratic objective to the Gurobi model @p QuadModel.
{
	if(this->made_yQy) return 0;
	GRBVar y[this->Ny];
	for (unsigned int i=0; i<Ny; i++)
		y[i] = this->QuadModel.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "y_"+to_string(i));
	GRBQuadExpr yQy{0};
	for(auto val=Q.begin(); val!=Q.end(); ++val)
	{
		unsigned int i, j;
		double value = (*val);
		i=val.row(); j=val.col();
		yQy += 0.5*y[i]*value*y[j];
	}
	QuadModel.setObjective(yQy);
	this->made_yQy = true;
	return 0;
}

unique_ptr<GRBModel> 
Game::QP_Param::solveFixed(arma::vec x ///< Other players' decisions
		)
/**
 * Given a value for the parameters @f$x@f$ in the definition of QP_Param, solve the parameterized quadratic program to  optimality. 
 *
 * In terms of game theory, this can be viewed as <i>the best response</i> for a set of decisions by other players.
 * 
 */
{
	this->make_yQy();
	/// @throws GRBException if argument vector size is not compatible with the Game::QP_Param definition.
	if(x.size()!=this->Nx) throw "Invalid argument size: " + to_string(x.size()) + " != "+to_string(Nx);
	/// @warning Creates a GRBModel using dynamic memory. Should be freed by the caller.
	unique_ptr<GRBModel> model(new GRBModel(this->QuadModel));
	try
	{
		GRBQuadExpr yQy = model->getObjective();
		arma::vec Cx, Ax;
		Cx = this->C*x;
		Ax = this->A*x;
		GRBVar y[this->Ny];
		for (unsigned int i=0; i<this->Ny; i++)
		{
			y[i] = model->getVarByName("y_"+to_string(i));
			yQy += (Cx[i]+c[i])*y[i];
		}
		model->setObjective(yQy);
		for (unsigned int i=0; i<this->Ncons; i++)
		{
			GRBLinExpr LHS{0};
			for(auto j=B.begin_row(i); j!=B.end_row(i); ++j)
				LHS += (*j)*y[j.col()];
			model->addConstr(LHS, GRB_LESS_EQUAL, b[i]-Ax[i]);
		}
		model->optimize();
	}
	catch(const char* e) { cerr<<" Error in Game::QP_Param::solveFixed: "<<e<<endl; throw;}
	catch(string e) { cerr<<"String: Error in Game::QP_Param::solveFixed: "<<e<<endl; throw;}
	catch(exception &e) { cerr<<"Exception: Error in Game::QP_Param::solveFixed: "<<e.what()<<endl; throw;}
	catch(GRBException &e){cerr<<"GRBException: Error in Game::QP_Param::solveFixed: "<<e.getErrorCode()<<"; "<<e.getMessage()<<endl;throw;}
	return model;
}

Game::QP_Param& Game::QP_Param::addDummy(unsigned int pars, unsigned int vars)
/**
 * @warning You might have to rerun QP_Param::KKT since you have now changed the QP.
 * @warning This implies you might have to rerun NashGame::FormulateLCP again too.
 */
{
	if(VERBOSE && (pars||vars)) cout<<"From Game::QP_Param::addDummyVars:\t You might have to rerun Games::QP_Param::KKT since you have now changed the number of variables in the NashGame.\n";

	// Call the superclass function
	MP_Param::addDummy(pars, vars);

	return *this;
}

unsigned int 
Game::QP_Param::KKT(arma::sp_mat& M, arma::sp_mat& N, arma::vec& q) const
/// @brief Compute the KKT conditions for the given QP
/**
 * Writes the KKT condition of the parameterized QP
 * As per the convention, y is the decision variable for the QP and 
 * that is parameterized in x
 * The KKT conditions are
 * \f$0 \leq y \perp  My + Nx + q \geq 0\f$
*/
{
	if (!this->dataCheck())
	{
		throw string("Inconsistent data for KKT of Game::QP_Param");
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


Game::QP_Param& 
Game::QP_Param::set(arma::sp_mat &Q, arma::sp_mat &C, arma::sp_mat &A, arma::sp_mat &B, arma::vec &c, arma::vec &b)
/// Setting the data, while keeping the input objects intact
{
	this->made_yQy = false;
	try { MP_Param::set(Q, C, A, B, c, b); }
	catch(string &e) {cerr<<"String: "<<e<<endl; throw string("Error in QP_Param::set: Invalid Data");}
	return *this;
}



Game::QP_Param& 
Game::QP_Param::set(arma::sp_mat &&Q, arma::sp_mat &&C, arma::sp_mat &&A, arma::sp_mat &&B, arma::vec &&c, arma::vec &&b)
/// Faster means to set data. But the input objects might be corrupted now.
{
	this->made_yQy = false;
	try { MP_Param::set(Q, C, A, B, c, b); }
	catch(string &e) {cerr<<"String: "<<e<<endl; throw string("Error in QP_Param::set: Invalid Data");}
	return *this;
}


Game::NashGame::NashGame(vector<shared_ptr<QP_Param>> Players, arma::sp_mat MC, arma::vec MCRHS, unsigned int n_LeadVar, arma::sp_mat LeadA, arma::vec LeadRHS):LeaderConstraints{LeadA}, LeaderConsRHS{LeadRHS}
/**
 * @brief
 * Construct a NashGame by giving a vector of pointers to 
 * QP_Param, defining each player's game
 * A set of Market clearing constraints and its RHS
 * And if there are leader variables, the number of leader vars.
 */
/**
 * Have a vector of pointers to Game::QP_Param ready such that
 * the variables are separated in \f$x^{i}\f$ and \f$x^{-i}\f$
 * format.
 *
 * In the correct ordering of variables, have the 
 * Market clearing equations ready. 
 *
 * Now call this constructor.
 * It will allocate appropriate space for the dual variables 
 * for each player.
 *
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

	this->set_positions();
}

void Game::NashGame::set_positions()
/**
 * Stores the position of each players' primal and dual variables. Also allocates Leader's position appropriately.
 * The ordering is according to the columns of
	 @image html FormulateLCP.png
	 @image latex FormulateLCP.png
 */
{
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

const Game::NashGame&
Game::NashGame::FormulateLCP(
		arma::sp_mat &M,  ///< Where the output  M is stored and returned. 
		arma::vec &q, 		///< Where the output  q is stored and returned.
		perps &Compl, 		///< Says which equations are complementary to which variables
		bool writeToFile, 	///< If  true, writes  M and  q to file.k
		string M_name, 		///< File name to be used to write  M
		string q_name		///< File name to be used to write  M
		) const
{
/// @brief Formulates the LCP corresponding to the Nash game. 
/// @warning Does not return the leader constraints. Use NashGame::RewriteLeadCons() to handle them
/**
 * Computes the KKT conditions for each Player, calling QP_Param::KKT. Arranges them systematically to return M, q
 * as an LCP @f$0\leq q \perp Mx+q \geq 0 @f$.
 *
	 The way the variables of the players get distributed is shown in the image below
	 @image html FormulateLCP.png
	 @image latex FormulateLCP.png
 */

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
	if(this->MCRHS.n_elem >= 1) // It is possible that it is a Cournot game and there are no MC conditions!
	{
		M.submat(this->MC_dual_position,0,this->Leader_position-1,this->dual_position.at(0)-1) = this->MarketClearing;
		q.subvec(this->MC_dual_position,this->Leader_position-1) = -this->MCRHS;
		for(unsigned int j=this->MC_dual_position;j<this->Leader_position;j++)
			Compl.push_back({j, j}); 
	}
	if(writeToFile)
	{
		M.save(M_name, arma::arma_ascii);
		q.save(q_name, arma::arma_ascii);
	}
	return *this;
}

arma::sp_mat 
Game::NashGame::RewriteLeadCons() const
/** @brief Rewrites leader constraint adjusting for dual variables.
 * Rewrites leader constraints given earlier with added empty columns and spaces corresponding to Market clearing duals and other equation duals.
 * 
 * This becomes important if the Lower level complementarity problem is passed to LCP with upper level constraints.
 */
{
	arma::sp_mat A_in = this->LeaderConstraints;
	arma::sp_mat A_out;
	unsigned int  NvarLead{0};
	NvarLead = this->dual_position.back(); // Number of Leader variables (all variables)
	// NvarFollow = NvarLead - this->n_LeadVar;
	
	unsigned int n_Row, n_Col;
	n_Row = A_in.n_rows; n_Col = A_in.n_cols;
	A_out.zeros(n_Row, NvarLead);

	try
	{
		// Primal variables i.e., everything before MCduals are the same!
		A_out.cols(0, this->MC_dual_position-1)  = A_in.cols(0, this->MC_dual_position-1);
		A_out.cols(this->Leader_position, this->dual_position.at(0)-1) = A_in.cols(this->MC_dual_position, n_Col-1);
		return A_out;
	}
	catch(const char* e) { cerr<<"Error in NashGame::RewriteLeadCons: "<<e<<endl; throw;}
	catch(string e) { cerr<<"String: Error in NashGame::RewriteLeadCons: "<<e<<endl; throw;}
	catch(exception &e) { cerr<<"Exception: Error in NashGame::RewriteLeadCons: "<<e.what()<<endl; throw;}
	return A_in;
}

Game::NashGame& Game::NashGame::addDummy(unsigned int par)
{
	for(auto &q: this->Players)
		q->addDummy(par);
	
	this->n_LeadVar += par;
	if(this->LeaderConstraints.n_rows)
	{
		auto nnR = this->LeaderConstraints.n_rows;
		auto nnC = this->LeaderConstraints.n_cols;
		this->LeaderConstraints.resize(nnR , nnC+par);
	}
	this->set_positions();
	return *this;
}

Game::NashGame& Game::NashGame::addLeadCons(const arma::vec &a, double b)
{
	auto nC = this->LeaderConstraints.n_cols;
	if (a.n_elem != nC) throw string("Error in NashGame::addLeadCons: Leader constraint size incompatible");
	auto nR = this->LeaderConstraints.n_rows;
	this->LeaderConstraints.resize(nR+1, nC);
	// (static_cast<arma::mat>(a)).t();	// Apparently this is not reqd! a.t() already works in newer versions of armadillo
	LeaderConstraints.row(nR) = a.t(); 
	this->LeaderConsRHS.resize(nR+1);
	this->LeaderConsRHS(nR) = b;
	return *this;
}
