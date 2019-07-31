#include<iostream>
#include<memory>
#include<string>
#include"lcptolp.h"
#include<gurobi_c++.h>
#include<armadillo>
// #define VERBOSE true

using namespace std;

bool
operator==(vector<int> Fix1, vector<int> Fix2)
/**
 * @brief Checks if two vector<int> are of same size and hold same values in the same order
 * @warning Might be deprecated, as it pollutes global namespaces
 * @returns @p true if Fix1 and Fix2 have the same elements else @p false
 */
{
    if (Fix1.size() != Fix2.size()) return false;
    for (unsigned int i = 0; i < Fix1.size(); i++)
        if (Fix1[i] != Fix2[i]) return false;
    return true;
}

bool
operator<(vector<int> Fix1, vector<int> Fix2)
/**
 * @details \b GrandParent:
 *  	Either the same value as the grand child, or has 0 in that location
 *
 *  \b Grandchild:
 *  	Same val as grand parent in every location, except any val allowed, if grandparent is 0
 * @warning Might be deprecated, as it pollutes global namespaces
 * @returns @p true if Fix1 is (grand) child of Fix2
 */
{
    if (Fix1.size() != Fix2.size()) return false;
    for (unsigned int i = 0; i < Fix1.size(); i++)
        if (Fix1[i] != Fix2[i] && Fix1[i] * Fix2[i] != 0)
            return false; // Fix1 is not a child of Fix2
    return true;        // Fix1 is a child of Fix2
}


bool
operator>(vector<int> Fix1, vector<int> Fix2) {
    return (Fix2 < Fix1);
}

void
Game::LCP::defConst(GRBEnv *env)
/**
 * @brief Assign default values to LCP attributes
 * @details Internal member that can be called from multiple constructors
 * to assign default values to some attributes of the class.
 * @todo LCP::defConst can be replaced by a private constructor
 */
{
    AllPolyhedra = new vector<vector<short int> *>{};
    RelAllPol = new vector<vector<short int> *>{};
    Ai = new vector<arma::sp_mat *>{};
    bi = new vector<arma::vec *>{};
    if (VERBOSE)
        this->RlxdModel.set(GRB_IntParam_OutputFlag, 0);
    this->env = env;
    this->nR = this->M.n_rows;
    this->nC = this->M.n_cols;
}


Game::LCP::LCP(GRBEnv *env, ///< Gurobi environment required
               arma::sp_mat M,        ///< @p M in @f$Mx+q@f$
               arma::vec q,        ///< @p q in @f$Mx+q@f$
               perps Compl,        ///< Pairing equations and variables for complementarity
               arma::sp_mat A,        ///< Any equations without a complemntarity variable
               arma::vec b            ///< RHS of equations without complementarity variables
) : M{M}, q{q}, _A{A}, _b{b}, RlxdModel(*env)
/// @brief Constructor with M, q, compl pairs
{
    defConst(env);
    this->Compl = perps(Compl);
    sort(Compl.begin(), Compl.end(),
         [](pair<unsigned int, unsigned int> a, pair<unsigned int, unsigned int> b) { return a.first < b.first; }
    );
    for (auto p:Compl)
        if (p.first != p.second) {
            this->LeadStart = p.first;
            this->LeadEnd = p.second - 1;
            this->nLeader = this->LeadEnd - this->LeadStart + 1;
            this->nLeader = this->nLeader > 0 ? this->nLeader : 0;
            break;
        }
}

Game::LCP::LCP(GRBEnv *env,                ///< Gurobi environment required
               arma::sp_mat M,                 ///< @p M in @f$Mx+q@f$
               arma::vec q,                    ///< @p q in @f$Mx+q@f$
               unsigned int LeadStart,         ///< Position where variables which are not complementary to any equation starts
               unsigned LeadEnd,               ///< Position where variables which are not complementary to any equation ends
               arma::sp_mat A,                 ///< Any equations without a complemntarity variable
               arma::vec b                     ///< RHS of equations without complementarity variables
) : M{M}, q{q}, _A{A}, _b{b}, RlxdModel(*env)
/// @brief Constructor with M,q,leader posn
/**
 * @warning This might be deprecated to support LCP functioning without sticking to the output format of NashGame
 */
{
    defConst(env);
    this->LeadStart = LeadStart;
    this->LeadEnd = LeadEnd;
    this->nLeader = this->LeadEnd - this->LeadStart + 1;
    this->nLeader = this->nLeader > 0 ? this->nLeader : 0;
    for (unsigned int i = 0; i < M.n_rows; i++) {
        unsigned int count = i < LeadStart ? i : i + nLeader;
        Compl.push_back({i, count});
    }
}

Game::LCP::LCP(GRBEnv *env, const NashGame &N) : RlxdModel(*env)
/**
 *	@brief Constructer given a NashGame
 *	@details Given a NashGame, computes the KKT of the lower levels, and makes the appropriate LCP object.
 *
 *	This constructor is the most suited for highlevel usage.
 *	@note Most preferred constructor for user interface.
 */
{
    arma::sp_mat M;
    arma::vec q;
    perps Compl;
    N.FormulateLCP(M, q, Compl);
    LCP(env, M, q, Compl, N.RewriteLeadCons(), N.getMCLeadRHS());


    // This is a constructor code! Remember to delete
    /// @todo Delete the below section of code
    this->M = M;
    this->q = q;
    this->_A = N.RewriteLeadCons();
    this->_b = N.getMCLeadRHS();
    defConst(env);
    this->Compl = perps(Compl);
    sort(Compl.begin(), Compl.end(),
         [](pair<unsigned int, unsigned int> a, pair<unsigned int, unsigned int> b) { return a.first < b.first; }
    );
    for (auto p:Compl)
        if (p.first != p.second) {
            this->LeadStart = p.first;
            this->LeadEnd = p.second - 1;
            this->nLeader = this->LeadEnd - this->LeadStart + 1;
            this->nLeader = this->nLeader > 0 ? this->nLeader : 0;
            break;
        }
    // Delete no more!
}

Game::LCP::~LCP()
/** @brief Destructor of LCP */
/** LCP object owns the pointers to definitions of its polyhedra that it owns
 It has to be deleted and freed. */
{
    for (auto p:*(this->AllPolyhedra)) delete p;
    for (auto p:*(this->RelAllPol)) delete p;
    delete this->AllPolyhedra;
    delete this->RelAllPol;
    for (auto a:*(this->Ai)) delete a;
    for (auto b:*(this->bi)) delete b;
    delete Ai;
    delete bi;
}

void
Game::LCP::makeRelaxed()
/** @brief Makes a Gurobi object that relaxes complementarity constraints in an LCP */
/** @details A Gurobi object is stored in the LCP object, that has all complementarity constraints removed.
 * A copy of this object is used by other member functions */
{
    try {
        if (this->madeRlxdModel) return;
        GRBVar x[nC], z[nR];
        if (VERBOSE) cout << "In LCP::makeRelaxed(): " << nR << " " << nC << endl;
        for (unsigned int i = 0; i < nC; i++)
            x[i] = RlxdModel.addVar(0, GRB_INFINITY, 1, GRB_CONTINUOUS, "x_" + to_string(i));
        for (unsigned int i = 0; i < nR; i++)
            z[i] = RlxdModel.addVar(0, GRB_INFINITY, 1, GRB_CONTINUOUS, "z_" + to_string(i));
        for (unsigned int i = 0; i < nR; i++) {
            GRBLinExpr expr = 0;
            for (auto v = M.begin_row(i); v != M.end_row(i); ++v)
                expr += (*v) * x[v.col()];
            expr += q(i);
            RlxdModel.addConstr(expr, GRB_EQUAL, z[i], "z_" + to_string(i) + "_def");
        }
        // If @f$Ax \leq b@f$ constraints are there, they should be included too!
        if (this->_A.n_nonzero != 0 && this->_b.n_rows != 0) {
            if (_A.n_cols != nC || _A.n_rows != _b.n_rows) {
                cout << "(" << _A.n_rows << "," << _A.n_cols << ")\t" << _b.n_rows << " " << nC << endl;
                throw string("A and b are incompatible! Thrown from makeRelaxed()");
            }
            for (unsigned int i = 0; i < _A.n_rows; i++) {
                GRBLinExpr expr = 0;
                for (auto a = _A.begin_row(i); a != _A.end_row(i); ++a)
                    expr += (*a) * x[a.col()];
                RlxdModel.addConstr(expr, GRB_LESS_EQUAL, _b(i), "commonCons_" + to_string(i));
            }
        }
        RlxdModel.update();
        this->madeRlxdModel = true;
    }
    catch (const char *e) {
        cerr << "Error in Game::LCP::makeRelaxed: " << e << endl;
        throw;
    }
    catch (string e) {
        cerr << "String: Error in Game::LCP::makeRelaxed: " << e << endl;
        throw;
    }
    catch (exception &e) {
        cerr << "Exception: Error in Game::LCP::makeRelaxed: " << e.what() << endl;
        throw;
    }
    catch (GRBException &e) {
        cerr << "GRBException: Error in Game::LCP::makeRelaxed: " << e.getErrorCode() << "; " << e.getMessage() << endl;
        throw;
    }
}

unique_ptr<GRBModel>
Game::LCP::LCP_Polyhed_fixed(
        vector<unsigned int> FixEq,            ///< If index is present, equality imposed on that variable
        vector<unsigned int> FixVar            ///< If index is present, equality imposed on that equation
)
/** 
 * The returned model has constraints
 * corresponding to the indices in FixEq set to equality
 * and variables corresponding to the indices
 * present in FixVar set to equality (=0)
 * @note This model returned could either be a relaxation or a restriction or neither. If every index is present in at least one of the two vectors --- @p FixEq or @p FixVar --- then it is a restriction.
 * @note <tt>LCP::LCP_Polyhed_fixed({},{})</tt> is equivalent to accessing LCP::RlxdModel
 * @warning The FixEq and FixVar variables are used under a different convention here!
 * @warning Note that the model returned by this function has to be explicitly deleted using the delete operator.
 * @returns unique pointer to a GRBModel
 */
{
    makeRelaxed();
    unique_ptr<GRBModel> model(new GRBModel(this->RlxdModel));
    for (auto i:FixEq) {
        if (i >= nR) throw "Element in FixEq is greater than nC";
        model->getVarByName("z_" + to_string(i)).set(GRB_DoubleAttr_UB, 0);
    }
    for (auto i:FixVar) {
        if (i >= nC) throw "Element in FixEq is greater than nC";
        model->getVarByName("z_" + to_string(i)).set(GRB_DoubleAttr_UB, 0);
    }
    return model;
}

unique_ptr<GRBModel>
Game::LCP::LCP_Polyhed_fixed(
        arma::Col<int> FixEq,            ///< If non zero, equality imposed on variable
        arma::Col<int> FixVar            ///< If non zero, equality imposed on equation
)
/**
 * Returs a model created from a given model
 * The returned model has constraints
 * corresponding to the non-zero elements of FixEq set to equality
 * and variables corresponding to the non-zero
 * elements of FixVar set to equality (=0)
 * @note This model returned could either be a relaxation or a restriction or neither.  If FixEq + FixVar is at least 1 (element-wise), then it is a restriction.
 * @note <tt>LCP::LCP_Polyhed_fixed({0,...,0},{0,...,0})</tt> is equivalent to accessing LCP::RlxdModel
 * @warning Note that the model returned by this function has to be explicitly deleted using the delete operator.
 * @returns unique pointer to a GRBModel
 */
{
    makeRelaxed();
    unique_ptr<GRBModel> model{new GRBModel(this->RlxdModel)};
    for (unsigned int i = 0; i < nC; i++)
        if (FixVar[i])
            model->getVarByName("x_" + to_string(i)).set(GRB_DoubleAttr_UB, 0);
    for (unsigned int i = 0; i < nR; i++)
        if (FixEq[i])
            model->getVarByName("z_" + to_string(i)).set(GRB_DoubleAttr_UB, 0);
    model->update();
    return model;
}

unique_ptr<GRBModel>
Game::LCP::LCPasMIP(
        vector<short int> Fixes, ///< For each Variable, +1 fixes the equation to equality and -1 fixes the variable to equality. A value of 0 fixes neither.
        bool solve ///< Whether the model is to be solved before returned
)
/**
 * Uses the big M method to solve the complementarity problem. The variables and eqns to be set to equality can be given in Fixes in 0/+1/-1 notation
 * @note Returned model is \e always a restriction. For <tt>Fixes = {0,...,0}</tt>, the returned model would solve the exact LCP (up to bigM caused restriction).
 * @throws string if <tt> Fixes.size()!= </tt> number of equations (for complementarity).
 * @warning Note that the model returned by this function has to be explicitly deleted using the delete operator.
 * @returns unique pointer to a GRBModel
 */
{
    if (Fixes.size() != this->nR) throw string("Bad size for Fixes in Game::LCP::LCPasMIP");
    vector<unsigned int> FixVar, FixEq;
    for (unsigned int i = 0; i < nR; i++) {
        if (Fixes[i] == 1) FixEq.push_back(i);
        if (Fixes[i] == -1) FixVar.push_back(i > this->LeadStart ? i + this->nLeader : i);
    }
    return this->LCPasMIP(FixEq, FixVar, solve);
}


unique_ptr<GRBModel>
Game::LCP::LCPasMIP(
        vector<unsigned int> FixEq,    ///< If any equation is to be fixed to equality
        vector<unsigned int> FixVar, ///< If any variable is to be fixed to equality
        bool solve ///< Whether the model should be solved in the function before returned.
)
/**
 * Uses the big M method to solve the complementarity problem. The variables and eqns to be set to equality can be given in FixVar and FixEq.
 * @note Returned model is \e always a restriction. For <tt>FixEq = FixVar = {}</tt>, the returned model would solve the exact LCP (up to bigM caused restriction).
 * @warning Note that the model returned by this function has to be explicitly deleted using the delete operator.
 * @returns unique pointer to a GRBModel
 */
{
    makeRelaxed();
    unique_ptr<GRBModel> model{new GRBModel(this->RlxdModel)};
    // Creating the model
    try {
        GRBVar x[nC], z[nR], u[nR], v[nR];
        // Get hold of the Variables and Eqn Variables
        for (unsigned int i = 0; i < nC; i++) x[i] = model->getVarByName("x_" + to_string(i));
        for (unsigned int i = 0; i < nR; i++) z[i] = model->getVarByName("z_" + to_string(i));
        // Define binary variables for bigM
        for (unsigned int i = 0; i < nR; i++) u[i] = model->addVar(0, 1, 0, GRB_BINARY, "u_" + to_string(i));
        if (this->useIndicators)
            for (unsigned int i = 0; i < nR; i++)
                v[i] = model->addVar(0, 1, 0, GRB_BINARY, "v_" + to_string(i));
        // Include ALL Complementarity constraints using bigM
        if (VERBOSE) {
            if (this->useIndicators) { cout << "Using indicator constraints for complementarities." << endl; }
            else { cout << "Using bigM for complementarities with M=" << this->bigM << endl; }
        }
        GRBLinExpr expr = 0;
        for (auto p:Compl) {
            // z[i] <= Mu constraint

            // u[j]=0 --> z[i] <=0
            if (!this->useIndicators) {
                expr = bigM * u[p.first];
                model->addConstr(expr, GRB_GREATER_EQUAL, z[p.first],
                                 "z" + to_string(p.first) + "_L_Mu" + to_string(p.first));
            } else
                model->addGenConstrIndicator(u[p.first], 0, z[p.first], GRB_LESS_EQUAL, 0,
                                             "z_ind_" + to_string(p.first) + "_L_Mu_" + to_string(p.first));
            // x[i] <= M(1-u) constraint
            if (!this->useIndicators) {
                expr = bigM;
                expr -= bigM * u[p.first];
                model->addConstr(expr, GRB_GREATER_EQUAL, x[p.second],
                                 "x" + to_string(p.first) + "_L_MuDash" + to_string(p.first));
            } else
                model->addGenConstrIndicator(v[p.first], 1, x[p.second], GRB_LESS_EQUAL, 0,
                                             "x_ind_" + to_string(p.first) + "_L_MuDash_" + to_string(p.first));

            if (this->useIndicators)
                model->addConstr(u[p.first] + v[p.first], GRB_EQUAL, 1, "uv_sum_" + to_string(p.first));
        }
        // If any equation or variable is to be fixed to zero, that happens here!
        for (auto i:FixVar) model->addConstr(x[i], GRB_EQUAL, 0.0);
        for (auto i:FixEq) model->addConstr(z[i], GRB_EQUAL, 0.0);
        model->update();
        if (VERBOSE)
            cout << "IntegerTol=" << this->eps_int << ";FeasabilityTol=OptimalityTol=" << this->eps << endl;
        model->set(GRB_DoubleParam_IntFeasTol, this->eps_int);
        model->set(GRB_DoubleParam_FeasibilityTol, this->eps);
        model->set(GRB_DoubleParam_OptimalityTol, this->eps);

        if (solve) model->optimize();
        return model;
    }
    catch (const char *e) {
        cerr << "Error in Game::LCP::LCPasMIP: " << e << endl;
        throw;
    }
    catch (string e) {
        cerr << "String: Error in Game::LCP::LCPasMIP: " << e << endl;
        throw;
    }
    catch (exception &e) {
        cerr << "Exception: Error in Game::LCP::LCPasMIP: " << e.what() << endl;
        throw;
    }
    catch (GRBException &e) {
        cerr << "GRBException: Error in Game::LCP::LCPasMIP: " << e.getErrorCode() << "; " << e.getMessage() << endl;
        throw;
    }
    return nullptr;
}

bool
Game::LCP::errorCheck(
        bool throwErr    ///< If this is true, function throws an error, else, it just returns false
) const
/**
 * Checks if the `M` and `q` given to create the LCP object are of 
 * compatible size, given the number of leader variables
 */
{

    const unsigned int nR = M.n_rows;
    const unsigned int nC = M.n_cols;
    if (throwErr) {
        if (nR != q.n_rows) throw "M and q have unequal number of rows";
        if (nR + nLeader != nC)
            throw "Inconsistency between number of leader vars " + to_string(nLeader) + ", number of rows " +
                  to_string(nR) + " and number of cols " + to_string(nC);
    }
    return (nR == q.n_rows && nR + nLeader == nC);
}


void
Game::LCP::print(string end) {
    cout << "LCP with " << this->nR << " rows and " << this->nC << " columns." << end;
}

int Game::ConvexHull(
        const vector<arma::sp_mat *> *Ai,    ///< Inequality constraints LHS that define polyhedra whose convex hull is to be found
        const vector<arma::vec *> *bi,    ///< Inequality constraints RHS that define polyhedra whose convex hull is to be found
        arma::sp_mat &A,            ///< Pointer to store the output of the convex hull LHS
        arma::vec &b,                ///< Pointer to store the output of the convex hull RHS
        const arma::sp_mat Acom,            ///< any common constraints to all the polyhedra - lhs.
        const arma::vec bcom                ///< Any common constraints to ALL the polyhedra - RHS.
)
/** @brief Computing convex hull of finite unioon of polyhedra
 * @details Computes the convex hull of a finite union of polyhedra where 
 * each polyhedra @f$P_i@f$ is of the form
 * @f{eqnarray}{
 * A^ix &\leq& b^i\\
 * x &\geq& 0
 * @f}
 * This uses Balas' approach to compute the convex hull.
 *
 * <b>Cross reference:</b> Conforti, Michele; Cornuéjols, Gérard; and Zambelli, Giacomo. Integer programming. Vol. 271. Berlin: Springer, 2014. Refer: Eqn 4.31
*/
{
    // Count number of polyhedra and the space we are in!
    unsigned int nPoly{static_cast<unsigned int>(Ai->size())};
    unsigned int nC{static_cast<unsigned int>(Ai->front()->n_cols)};

    // Count the number of variables in the convex hull.
    unsigned int nFinCons{0}, nFinVar{0};
    // Error check
    if (nPoly == 0)
        throw string("Empty vector of polyhedra given!");    // There should be at least 1 polyhedron to consider
    if (nPoly != bi->size()) throw string("Inconsistent number of LHS and RHS for polyhedra");
    for (unsigned int i = 0; i != nPoly; i++) {
        if (Ai->at(i)->n_cols != nC)
            throw string("Inconsistent number of variables in the polyhedra ") + to_string(i) + "; " +
                  to_string(Ai->at(i)->n_cols) + "!=" + to_string(nC);
        if (Ai->at(i)->n_rows != bi->at(i)->n_rows)
            throw string("Inconsistent number of rows in LHS and RHS of polyhedra ") + to_string(i) + ";" +
                  to_string(Ai->at(i)->n_rows) + "!=" + to_string(bi->at(i)->n_rows);
        nFinCons += Ai->at(i)->n_rows;
    }
    unsigned int FirstCons = nFinCons;
    if (Acom.n_rows > 0 && Acom.n_cols != nC) throw string("Inconsistent number of variables in the common polyhedron");
    if (Acom.n_rows > 0 && Acom.n_rows != bcom.n_rows)
        throw string("Inconsistent number of rows in LHS and RHS in the common polyhedron");

    // 2nd constraint in Eqn 4.31 of Conforti - twice so we have 2 ineq instead of 1 eq constr
    nFinCons += nC * 2;
    // 3rd constr in Eqn 4.31. Again as two ineq constr.
    nFinCons += 2;
    // Common constraints
    // nFinCons += Acom.n_rows;

    nFinVar = nPoly * nC + nPoly + nC; // All x^i variables + delta variables+ original x variables
    A.zeros(nFinCons, nFinVar);
    b.zeros(nFinCons);
    // A.zeros(nFinCons, nFinVar); b.zeros(nFinCons);
    // Implements the first constraint more efficiently using better constructors for sparse matrix
    Game::compConvSize(A, nFinCons, nFinVar, Ai, bi);

    // Counting rows completed
    if (VERBOSE) { cout << "In Convex Hull computation!" << endl; }
    /****************** SLOW LOOP BEWARE *******************/
    for (unsigned int i = 0; i < nPoly; i++) {
        if (VERBOSE) cout << "Game::ConvexHull: Handling Polyhedron " << i + 1 << " out of " << nPoly << endl;
        // First constraint in (4.31)
        // A.submat(complRow, i*nC, complRow+nConsInPoly-1, (i+1)*nC-1) = *Ai->at(i); // Slowest line. Will arma improve this?
        // First constraint RHS
        // A.submat(complRow, nPoly*nC+i, complRow+nConsInPoly-1, nPoly*nC+i) = -*bi->at(i);
        // Second constraint in (4.31)
        for (unsigned int j = 0; j < nC; j++) {
            A.at(FirstCons + 2 * j, nC + (i * nC) + j) = 1;
            A.at(FirstCons + 2 * j + 1, nC + (i * nC) + j) = -1;
        }
        // Third constraint in (4.31)
        A.at(FirstCons + nC * 2, nC + nPoly * nC + i) = 1;
        A.at(FirstCons + nC * 2 + 1, nC + nPoly * nC + i) = -1;
    }
    /****************** SLOW LOOP BEWARE *******************/
    // Second Constraint RHS
    for (unsigned int j = 0; j < nC; j++) {
        A.at(FirstCons + 2 * j, j) = -1;
        A.at(FirstCons + 2 * j + 1, j) = 1;
    }
    // Third Constraint RHS
    b.at(FirstCons + nC * 2) = 1;
    b.at(FirstCons + nC * 2 + 1) = -1;
    // Common Constraints
    if (Acom.n_rows > 0) {
        arma::sp_mat A_comm_temp;
        A_comm_temp = arma::join_rows(Acom, arma::zeros<arma::sp_mat>(Acom.n_rows, nFinVar - nC));
        A = arma::join_cols(A, A_comm_temp);
        b = arma::join_cols(b, bcom);
        // A = arma::join_cols(A_comm_temp, A); b = arma::join_cols(bcom, b);

        /*		b.subvec(FirstCons+2*nC+2, nFinCons-1) = bcom;
               A.submat(FirstCons+2*nC+2, 0, //  nPoly*nC+nPoly,
                           nFinCons-1, nC-1 // nFinVar-1
                           ) = Acom;*/
    }
    if (VERBOSE) cout << "Convex Hull A:" << A.n_rows << "x" << A.n_cols << endl;
    return 0;
}


void Game::compConvSize(arma::sp_mat &A,    ///< Output parameter
                        const unsigned int nFinCons,            ///< Number of rows in final matrix A
                        const unsigned int nFinVar,            ///< Number of columns in the final matrix A
                        const vector<arma::sp_mat *> *Ai,    ///< Inequality constraints LHS that define polyhedra whose convex hull is to be found
                        const vector<arma::vec *> *bi    ///< Inequality constraints RHS that define polyhedra whose convex hull is to be found
)
/**
 * @brief INTERNAL FUNCTION NOT FOR GENERAL USE.
 * @warning INTERNAL FUNCTION NOT FOR GENERAL USE.
 * @internal To generate the matrix "A" in Game::ConvexHull using batch insertion constructors. This is faster than the original line in the code:
 * A.submat(complRow, i*nC, complRow+nConsInPoly-1, (i+1)*nC-1) = *Ai->at(i);
 * Motivation behind this: Response from armadillo:-https://gitlab.com/conradsnicta/armadillo-code/issues/111
 */
{
    unsigned int nPoly{static_cast<unsigned int>(Ai->size())};
    unsigned int nC{static_cast<unsigned int>(Ai->front()->n_cols)};
    unsigned int N{0}; // Total number of nonzero elements in the final matrix
    for (unsigned int i = 0; i < nPoly; i++) {
        N += Ai->at(i)->n_nonzero;
        N += bi->at(i)->n_rows;
    }

    // Now computed N which is the total number of nonzeros.
    arma::umat locations;    // location of nonzeros
    arma::vec val;            // nonzero values
    locations.zeros(2, N);
    val.zeros(N);

    unsigned int count{0}, rowCount{0}, colCount{nC};
    for (unsigned int i = 0; i < nPoly; i++) {
        for (auto it = Ai->at(i)->begin(); it != Ai->at(i)->end(); ++it) // First constraint
        {
            locations(0, count) = rowCount + it.row();
            locations(1, count) = colCount + it.col();
            val(count) = *it;
            ++count;
        }
        for (unsigned int j = 0; j < bi->at(i)->n_rows; ++j) // RHS of first constraint
        {
            locations(0, count) = rowCount + j;
            locations(1, count) = nC + nC * nPoly + i;
            val(count) = -bi->at(i)->at(j);
            ++count;
        }
        colCount += nC;
        rowCount += Ai->at(i)->n_rows;
        if (VERBOSE) cout << "In compConvSize: " << i + 1 << " out of " << nPoly << endl;
    }
    A = arma::sp_mat(locations, val, nFinCons, nFinVar);
}

arma::vec
Game::LPSolve(const arma::sp_mat &A, ///< The constraint matrix
              const arma::vec &b,        ///< RHS of the constraint matrix
              const arma::vec &c,        ///< If feasible, returns a vector that minimizes along this direction
              int &status,                ///< Status of the optimization problem. If optimal, this will be GRB_OPTIMAL
              bool Positivity                ///< Should @f$x\geq0@f$ be enforced?
)
/**
 Checks if the polyhedron given by @f$ Ax\leq b@f$ is feasible.
 If yes, returns the point @f$x@f$ in the polyhedron that minimizes @f$c^Tx@f$
 Positivity can be enforced on the variables easily.
*/
{
    unsigned int nR, nC;
    nR = A.n_rows;
    nC = A.n_cols;
    if (c.n_rows != nC) throw "Inconsistency in no of Vars in isFeas()";
    if (b.n_rows != nR) throw "Inconsistency in no of Constr in isFeas()";

    arma::vec sol = arma::vec(c.n_rows, arma::fill::zeros);
    const double lb = Positivity ? 0 : -GRB_INFINITY;

    GRBEnv env;
    GRBModel model = GRBModel(env);
    GRBVar x[nC];
    GRBConstr a[nR];
    // Adding Variables
    for (unsigned int i = 0; i < nC; i++)
        x[i] = model.addVar(lb, GRB_INFINITY, c.at(i), GRB_CONTINUOUS, "x_" + to_string(i));
    // Adding constraints
    for (unsigned int i = 0; i < nR; i++) {
        GRBLinExpr lin{0};
        for (auto j = A.begin_row(i); j != A.end_row(i); ++j)
            lin += (*j) * x[j.col()];
        a[i] = model.addConstr(lin, GRB_LESS_EQUAL, b.at(i));
    }
    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_IntParam_DualReductions, 0);
    model.optimize();
    status = model.get(GRB_IntAttr_Status);
    if (status == GRB_OPTIMAL)
        for (unsigned int i = 0; i < nC; i++) sol.at(i) = x[i].get(GRB_DoubleAttr_X);
    return sol;
}


vector<short int> *
Game::LCP::anyBranch(const vector<vector<short int> *> *vecOfFixes, vector<short int> *Fix) const
/** @brief Returns the (grand)child if any (grand)child of @p Fix is in @p vecOfFixes else returns @c nullptr  */
/**
 *  \b GrandParent:
 *  	Either the same value as the grand child, or has 0 in that location
 *
 *  \b Grandchild:
 *  	Same val as grand parent in every location, except any val allowed, if grandparent is 0
 */
{
    for (auto v:*vecOfFixes)
        if (*Fix < *v || *v == *Fix) return v;
    return NULL;
}

bool
Game::LCP::extractSols(
        GRBModel *model,    ///< The Gurobi Model that was solved (perhaps using Game::LCP::LCPasMIP)
        arma::vec &z,        ///< Output variable - where the equation values are stored
        arma::vec &x,        ///< Output variable - where the variable values are stored
        bool extractZ        ///< z values are filled only if this is true
) const
/** @brief Extracts variable and equation values from a solved Gurobi model for LCP */
/** @warning This solves the model if the model is not already solve */
/** @returns @p false if the model is not solved to optimality. @p true otherwise */
{
    if (model->get(GRB_IntAttr_Status) == GRB_LOADED) model->optimize();
    if (model->get(GRB_IntAttr_Status) != GRB_OPTIMAL) return false;
    x.zeros(nC);
    if (extractZ) z.zeros(nR);
    for (unsigned int i = 0; i < nR; i++) {
        x[i] = model->getVarByName("x_" + to_string(i)).get(GRB_DoubleAttr_X);
        if (extractZ) z[i] = model->getVarByName("z_" + to_string(i)).get(GRB_DoubleAttr_X);
    }
    for (unsigned int i = nR; i < nC; i++)
        x[i] = model->getVarByName("x_" + to_string(i)).get(GRB_DoubleAttr_X);
    return true;
}

vector<short int> *
Game::LCP::solEncode(const arma::vec &z, ///< Equation values
                     const arma::vec &x                 ///< Variable values
) const
/// @brief Given variable values and equation values, encodes it in 0/+1/-1 format and returns it.
/// @warning Note that the vector returned by this function might have to be explicitly deleted using the delete operator. For specific uses in LCP::BranchAndPrune, this delete is handled by the class destructor.
{
    vector<signed short int> *solEncoded = new vector<signed short int>(nR, 0);
    for (auto p:Compl) {
        unsigned int i, j;
        i = p.first;
        j = p.second;
        if (isZero(z(i))) solEncoded->at(i)++;
        if (isZero(x(j))) solEncoded->at(i)--;
    };
    return solEncoded;
}

vector<short int> *
Game::LCP::solEncode(GRBModel *model) const
/// @brief Given a Gurobi model, extracts variable values and equation values, encodes it in 0/+1/-1 format and returns it.
/// @warning Note that the vector returned by this function might have to be explicitly deleted using the delete operator. For specific uses in LCP::BranchAndPrune, this delete is handled by the class destructor.
{
    arma::vec x, z;
    if (!this->extractSols(model, z, x, true)) return {};// If infeasible model, return empty!
    else return this->solEncode(z, x);
}

void
Game::LCP::branch(int loc,                    ///< Location (complementarity pair) to be branched at.
                  const vector<short int> *Fixes)        ///< What are fixed so far.

/** @brief Branches at a location defined by the caller
 * If loc == nR, then stop branching. We either hit infeasibility or a leaf (i.e., all locations are branched)
 * If loc <0, then branch at abs(loc) location and go down the branch where variable is fixed to 0
 * else branch at abs(loc) location and go down the branch where eqn is fixed to 0
 * @note This is just a handler for Branch and bound and obeys what other functions like LCP::branchLoc and LCP::branchProcLoc suggest it to do. No major change expected here.
 */
{
    bool VarFirst = (loc < 0);
    unique_ptr<GRBModel> FixEqMdl, FixVarMdl;
    // GRBModel *FixEqMdl=nullptr, *FixVarMdl=nullptr;
    vector<short int> *FixEqLeaf, *FixVarLeaf;

    if (VERBOSE) {
        cout << "\nBranching on Variable: " << loc << " with Fix as ";
        for (auto t1:*Fixes) cout << t1 << '\t';
        cout << endl;
    }

    loc = (loc >= 0) ? loc : (loc == -(int) nR ? 0 : -loc);
    if (loc >= static_cast<signed int>(nR)) {
        if (VERBOSE) cout << "nR: " << nR << "\tloc: " << loc << "\t Returning..." << endl;
        return;
    } else {
        GRBVar x, z;
        vector<short int> *FixesEq = new vector<short int>(*Fixes);
        vector<short int> *FixesVar = new vector<short int>(*Fixes);
        if (Fixes->at(loc) != 0) throw string("Error in LCP::branch: Fixing an already fixed variable!");
        FixesEq->at(loc) = 1;
        FixesVar->at(loc) = -1;
        if (VarFirst) {
            FixVarLeaf = anyBranch(AllPolyhedra,
                                   FixesVar); // Checking if a feasible solution is already found along this branch
            if (!FixVarLeaf) this->branch(branchLoc(FixVarMdl, FixesVar), FixesVar);
            else this->branch(branchProcLoc(FixesVar, FixVarLeaf), FixesVar);

            FixEqLeaf = anyBranch(AllPolyhedra, FixesEq);
            if (!FixEqLeaf) this->branch(branchLoc(FixEqMdl, FixesEq), FixesEq);
            else this->branch(branchProcLoc(FixesEq, FixEqLeaf), FixesEq);
        } else {
            FixEqLeaf = anyBranch(AllPolyhedra, FixesEq);
            if (!FixEqLeaf) this->branch(branchLoc(FixEqMdl, FixesEq), FixesEq);
            else this->branch(branchProcLoc(FixesEq, FixEqLeaf), FixesEq);

            FixVarLeaf = anyBranch(AllPolyhedra, FixesVar);
            if (!FixVarLeaf) this->branch(branchLoc(FixVarMdl, FixesVar), FixesVar);
            else this->branch(branchProcLoc(FixesVar, FixVarLeaf), FixesVar);
        }
        delete FixesEq;
        delete FixesVar;
    }
}

vector<vector<short int> *> *
Game::LCP::BranchAndPrune()
/**
 * @brief Calls the complete branch and prune for LCP object.
 * @returns LCP::AllPolyhedra 
 */
{
    unique_ptr<GRBModel> m;
    vector<short int> *Fix = new vector<short int>(nR, 0);
    branch(branchLoc(m, Fix), Fix);
    delete Fix;
    return AllPolyhedra;
}

int
Game::LCP::branchLoc(unique_ptr<GRBModel> &m, vector<short int> *Fix)
/**
 * @brief Defining the branching rule.
 * @details Solves a gurobi model at some point in the branch and prune tree. From there, uses a heuristic to find the complementarity equation where the branching is to be done and information on whether the exploration is go inside the "equation" side of the branch or the "variable" side of the branch. Currently, the largest non-zero value is the value which would be forced to be zero (with the hope infeasibility would be identified fast).
 * @note This can be vastly improved with better branching rules.
 * @returns <tt>signed int</tt> that details where branching has to be done. A positive value implies, branching will be on equation, a negative value implies branching will be on variable. As an exception, 0 implies branching on first eqn and <tt> - LCP::nR</tt> implies branching on first equation. A return value of <tt>LCP::nR</tt> implies that the node is infeasible and that there is no point in branching.
 */
{
    static int GurCallCt{0};
    m = this->LCPasMIP(*Fix, true);
    GurCallCt++;
    if (VERBOSE) {
        cout << "Gurobi call\t" << GurCallCt << "\t";
        for (auto a:*Fix) cout << a << "\t";
        cout << endl;
    }
    int pos;
    pos = (signed int) nR;// Don't branch! You are at the leaf if pos never gets changed!!
    arma::vec z, x;
    if (this->extractSols(m.get(), z, x, true)) // If already infeasible, nothing to branch!
    {
        vector<short int> *v1 = this->solEncode(z, x);
        if (VERBOSE) {
            cout << "v1: \t\t\t";
            for (auto a:*v1) cout << a << '\t';
            cout << '\n';
        }

        this->AllPolyhedra->push_back(v1);
        this->FixToPolies(v1);

        if (VERBOSE) {
            cout << "New Polyhedron found" << endl;
            x.t().print("x");
            z.t().print("z");
        }
        ////////////////////
        // BRANCHING RULE //
        ////////////////////
        // Branch at a large positive value
        double maxvalx{0};
        unsigned int maxposx{nR};
        double maxvalz{0};
        unsigned int maxposz{nR};
        for (unsigned int i = 0; i < nR; i++) {
            unsigned int varPos = i >= this->LeadStart ? i + nLeader : i;
            if (x(varPos) > maxvalx && Fix->at(i) == 0) // If already fixed, it makes no sense!
            {
                maxvalx = x(varPos);
                maxposx = (i == 0) ? -nR : -i; // Negative of 0 is -nR by convention
            }
            if (z(i) > maxvalz && Fix->at(i) == 0) // If already fixed, it makes no sense!
            {
                maxvalz = z(i);
                maxposz = i;
            }
        }
        pos = maxvalz > maxvalx ? maxposz : maxposx;
        ///////////////////////////
        // END OF BRANCHING RULE //
        ///////////////////////////
    } else { if (VERBOSE) cout << "Infeasible branch" << endl; }
    return pos;
}

int
Game::LCP::branchProcLoc(vector<short int> *Fix, vector<short int> *Leaf)
/**
 * @brief Branching choice, if we are at a processed node
 * @details When at processed node, we know that the node definitely has a feasible descendent. 
 * This just finds the first unbranched complementarity equation and branches there. 
 * @return Branch location if not at leaf and LCP::nR if at leaf.
 */
{
    int pos = (int) nR;

    if (VERBOSE) {
        cout << "Processed Node \t\t";
        for (auto a:*Fix) cout << a << "\t";
        cout << endl;
    }

    if (*Fix == *Leaf) return nR;
    if (Fix->size() != Leaf->size()) throw "Error in branchProcLoc";
    for (unsigned int i = 0; i < Fix->size(); i++) {
        int l = Leaf->at(i);
        if (Fix->at(i) == 0) return (l == 0 ? -i : i * l);
    }
    return pos;
}

Game::LCP &
Game::LCP::FixToPoly(
        const vector<short int> *Fix,    ///< A vector of +1 and -1 referring to which equations and variables are taking 0 value.
        bool checkFeas,                                ///< The polyhedron is added after ensuring feasibility, if this is true
        bool custom,                                    ///< Should the polyhedra be pushed into a custom vector of polyhedra as opposed to LCP::Ai and LCP::bi
        vector<arma::sp_mat *> *custAi,                    ///< If custom polyhedra vector is used, pointer to vector of LHS constraint matrix
        vector<arma::vec *> *custbi                        /// If custom polyhedra vector is used, pointer to vector of RHS of constraints
)
/** @brief Computes the equation of the feasibility polyhedron corresponding to the given @p Fix
 *	@details The computed polyhedron is always pushed into a vector of @p arma::sp_mat and @p arma::vec 
 *	If @p custom is false, this is the internal attribute of LCP, which are LCP::Ai and LCP::bi.
 *	Otherwise, the vectors can be provided as arguments.
 *	@p true value to @p checkFeas ensures that the polyhedron is pushed @e only if it is feasible.
 *	@warning Does not entertain 0 in the elements of *Fix. Only +1/-1 are allowed to not encounter undefined behavior. As a result, 
 *	not meant for high level code. Instead use LCP::FixToPolies.
 */
{
    arma::sp_mat *Aii = new arma::sp_mat(nR, nC);
    arma::vec *bii = new arma::vec(nR, arma::fill::zeros);
    for (unsigned int i = 0; i < this->nR; i++) {
        if (Fix->at(i) == 0) throw string("Error in Game::LCP::FixToPoly. 0s not allowed in argument vector");
        if (Fix->at(i) == 1) // Equation to be fixed top zero
        {
            for (auto j = this->M.begin_row(i); j != this->M.end_row(i); ++j)
                if (!this->isZero((*j)))
                    Aii->at(i, j.col()) = (*j); // Only mess with non-zero elements of a sparse matrix!
            bii->at(i) = this->q(i);
        } else // Variable to be fixed to zero, i.e. x(j) <= 0 constraint to be added
        {
            unsigned int varpos = (i > this->LeadStart) ? i + this->nLeader : i;
            Aii->at(i, varpos) = 1;
            bii->at(i) = 0;
        }
    }
    bool add = !checkFeas;
    if (checkFeas) {
        unsigned int count{0};
        try {
            makeRelaxed();
            GRBModel *model = new GRBModel(this->RlxdModel);
            for (auto i:*Fix) {
                if (i > 0) model->getVarByName("z_" + to_string(count)).set(GRB_DoubleAttr_UB, 0);
                if (i < 0)
                    model->getVarByName("x_" + to_string(count > this->LeadStart ? count + nLeader : i)).set(
                            GRB_DoubleAttr_UB, 0);
                count++;
            }
            model->optimize();
            if (model->get(GRB_IntAttr_Status) == GRB_OPTIMAL) add = true;
            delete model;
        }
        catch (const char *e) {
            cerr << "Error in Game::LCP::FixToPoly: " << e << endl;
            throw;
        }
        catch (string e) {
            cerr << "String: Error in Game::LCP::FixToPoly: " << e << endl;
            throw;
        }
        catch (exception &e) {
            cerr << "Exception: Error in Game::LCP::FixToPoly: " << e.what() << endl;
            throw;
        }
        catch (GRBException &e) {
            cerr << "GRBException: Error in Game::LCP::FixToPoly: " << e.getErrorCode() << ": " << e.getMessage()
                 << endl;
            throw;
        }
    }
    if (add) {
        custom ? (custAi->push_back(Aii)) : (this->Ai->push_back(Aii));
        custom ? custbi->push_back(bii) : this->bi->push_back(bii);
        if (VERBOSE) cout << custom << " Pushed a new polyhedron! No: " << custAi->size() << endl;
    }
    return *this;
}

Game::LCP &
Game::LCP::FixToPolies(
        const vector<short int> *Fix,    ///< A vector of +1, 0 and -1 referring to which equations and variables are taking 0 value.
        bool checkFeas,                                 ///< The polyhedron is added after ensuring feasibility, if this is true
        bool custom,                                    ///< Should the polyhedra be pushed into a custom vector of polyhedra as opposed to LCP::Ai and LCP::bi
        vector<arma::sp_mat *> *custAi,                  ///< If custom polyhedra vector is used, pointer to vector of LHS constraint matrix
        vector<arma::vec *> *custbi                      /// If custom polyhedra vector is used, pointer to vector of RHS of constraints
)
/** @brief Computes the equation of the feasibility polyhedron corresponding to the given @p Fix
 *	@details The computed polyhedron are always pushed into a vector of @p arma::sp_mat and @p arma::vec
 *	If @p custom is false, this is the internal attribute of LCP, which are LCP::Ai and LCP::bi.
 *	Otherwise, the vectors can be provided as arguments.
 *	@p true value to @p checkFeas ensures that @e each polyhedron that is pushed is feasible.
 *	not meant for high level code. Instead use LCP::FixToPolies.
 *	@note A value of 0 in @p *Fix implies that polyhedron corresponding to fixing the corresponding variable as well as the equation
 *	become candidates to pushed into the vector. Hence this is preferred over LCP::FixToPoly for high-level usage.
 */
{
    bool flag = false;
    vector<short int> MyFix(*Fix);
    if (VERBOSE) {
        for (const auto v:MyFix) cout << v << " ";
        cout << endl;
    }
    unsigned int i;
    for (i = 0; i < this->nR; i++) {
        if (Fix->at(i) == 0) {
            flag = true;
            break;
        }
    }
    if (flag) {
        MyFix[i] = 1;
        this->FixToPolies(&MyFix, checkFeas, custom, custAi, custbi);
        MyFix[i] = -1;
        this->FixToPolies(&MyFix, checkFeas, custom, custAi, custbi);
    } else this->FixToPoly(Fix, checkFeas, custom, custAi, custbi);
    return *this;
}

Game::LCP &
Game::LCP::EnumerateAll(const bool solveLP ///< Should the poyhedra added be checked for feasibility?
)
/**
 * @brief Brute force computation of LCP feasible region
 * @details Computes all @f$2^n@f$ polyhedra defining the LCP feasible region.
 * Th ese are always added to LCP::Ai and LCP::bi
 */
{
    delete Ai;
    delete bi; // Just in case it is polluted with BranchPrune
    Ai = new vector<arma::sp_mat *>{};
    bi = new vector<arma::vec *>{};
    vector<short int> *Fix = new vector<short int>(nR, 0);
    this->FixToPolies(Fix, solveLP);
    return *this;
}

Game::LCP &
Game::LCP::addPolyhedron(
        const vector<short int> &Fix,    ///< +1/0/-1 Representation of the polyhedra which needed to be pushed
        vector<arma::sp_mat *> &custAi,        ///< Vector with LHS of constraint matrix should be pushed.
        vector<arma::vec *> &custbi,            ///< Vector with RHS of constraints should be pushed.
        const bool convHull,                ///< If @p true convex hull of @e all polyhedra in custAi, custbi will be computed
        arma::sp_mat *A,                    ///< Location where convex hull LHS has to be stored
        arma::vec *b                        ///< Location where convex hull RHS has to be stored
) {
    this->FixToPolies(&Fix, false, true, &custAi, &custbi);
    if (convHull) {
        arma::sp_mat A_common;
        A_common = arma::join_cols(this->_A, -this->M);
        arma::vec b_common = arma::join_cols(this->_b, this->q);
        if (!this->convexify) {
            *A = A_common;
            *b = b_common;
            cout << "WARNING: Convexification is disabled." << endl;
        } else {
            Game::ConvexHull(&custAi, &custbi, *A, *b, A_common, b_common);
        }
    }
    return *this;
}

Game::LCP &
Game::LCP::makeQP(
        const vector<short int> &Fix,    ///< +1/0/-1 Representation of the polyhedron which needed to be pushed
        vector<arma::sp_mat *> &custAi,    ///< Vector with LHS of constraint matrix should be pushed.
        vector<arma::vec *> &custbi,        ///< Vector with RHS of constraints should be pushed.
        Game::QP_objective &QP_obj,        ///< The objective function of the QP to be returned. @warning Size of this parameter might change!
        Game::QP_Param &QP                ///< The output parameter where the final Game::QP_Param object is stored
) {
    // Original sizes
    const unsigned int Nx_old{static_cast<unsigned int>(QP_obj.C.n_cols)};


    if (VERBOSE) cout << QP_obj.C.n_cols << " " << QP_obj.C.n_rows << endl;
    Game::QP_constraints QP_cons;
    this->addPolyhedron(Fix, custAi, custbi, true, &QP_cons.B, &QP_cons.b);
    // Updated size after convex hull has been computed.
    const unsigned int Ncons{static_cast<unsigned int>(QP_cons.B.n_rows)};
    const unsigned int Ny{static_cast<unsigned int>(QP_cons.B.n_cols)};
    // Resizing entities.
    QP_cons.A.zeros(Ncons, Nx_old);
    QP_obj.c = resize_patch(QP_obj.c, Ny, 1);
    QP_obj.C = resize_patch(QP_obj.C, Ny, Nx_old);
    QP_obj.Q = resize_patch(QP_obj.Q, Ny, Ny);
    // Setting the QP_Param object
    QP.set(QP_obj, QP_cons);
    return *this;
}


Game::LCP &
Game::LCP::makeQP(
        Game::QP_objective &QP_obj,
        Game::QP_Param &QP
) {
    vector<arma::sp_mat *> custAi{};
    vector<arma::vec *> custbi{};
    vector<short int> Fix = vector<short int>(this->getCompl().size(), 0); // Complete enumeration
    return this->makeQP(Fix, custAi, custbi, QP_obj, QP);
}


unique_ptr<GRBModel>
Game::LCP::LCPasQP(bool solve)
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
    for (auto p:this->Compl) {
        unsigned int i = p.first;
        unsigned int j = p.second;
        z[i] = model->getVarByName("z_" + to_string(i));
        x[i] = model->getVarByName("x_" + to_string(j));
        obj += x[i] * z[i];
    }
    model->setObjective(obj, GRB_MINIMIZE);
    if (solve) {
        try {
            model->optimize();
            int status = model->get(GRB_IntAttr_Status);
            if (status != GRB_OPTIMAL || model->get(GRB_DoubleAttr_ObjVal) > this->eps)
                throw "LCP infeasible";
        }
        catch (const char *e) {
            cerr << "Error in Game::LCP::LCPasQP: " << e << endl;
            throw;
        }
        catch (string e) {
            cerr << "String: Error in Game::LCP::LCPasQP: " << e << endl;
            throw;
        }
        catch (exception &e) {
            cerr << "Exception: Error in Game::LCP::LCPasQP: " << e.what() << endl;
            throw;
        }
        catch (GRBException &e) {
            cerr << "GRBException: Error in Game::LCP::LCPasQP: " << e.getErrorCode() << "; " << e.getMessage() << endl;
            throw;
        }
    }
    return model;
}

unique_ptr<GRBModel>
Game::LCP::LCPasMIP(bool solve)
/**
 * @brief Helps solving an LCP as an MIP using bigM constraints
 * @returns A unique_ptr to GRBModel that has the equivalent MIP
 * @details The MIP problem that is returned by this function is equivalent to the LCP problem provided the value of bigM is large enough.
 * @note This solves just the feasibility problem. Should you need  a leader's objective function, use LCP::MPECasMILP or LCP::MPECasMIQP
 */
{
    return this->LCPasMIP({}, {}, solve);
}


unique_ptr<GRBModel>
Game::LCP::MPECasMILP(const arma::sp_mat &C, const arma::vec &c, const arma::vec &x_minus_i, bool solve)
/**
 * @brief Helps solving an LCP as an MIP using bigM constraints. 
 * @returns A unique_ptr to GRBModel that has the equivalent MIP
 * @details The MIP problem that is returned by this function is equivalent to the LCP problem provided the value of bigM is large enough. The function differs from LCP::LCPasMIP by the fact that, this explicitly takes a leader objective, and returns an object with this objective. 
 * @note The leader's objective has to be linear here. For quadratic objectives, refer LCP::MPECasMIQP
 */
{
    unique_ptr<GRBModel> model = this->LCPasMIP(false);
    arma::vec Cx(this->nC, arma::fill::zeros);
    try {
        Cx = C * x_minus_i;
        if (Cx.n_rows != this->nC) throw string("Bad size of C");
        if (c.n_rows != this->nC) throw string("Bad size of c");
    }
    catch (exception &e) {
        cerr << "Exception in Game::LCP::MPECasMIQP: " << e.what() << endl;
        throw;
    }
    catch (string &e) {
        cerr << "Exception in Game::LCP::MPECasMIQP: " << e << endl;
        throw;
    }
    arma::vec obj = c + Cx;
    GRBLinExpr expr{0};
    for (unsigned int i = 0; i < this->nC; i++)
        expr += obj.at(i) * model->getVarByName("x_" + to_string(i));
    model->setObjective(expr, GRB_MINIMIZE);
    if (solve) model->optimize();
    return model;
}

unique_ptr<GRBModel>
Game::LCP::MPECasMIQP(const arma::sp_mat &Q, const arma::sp_mat &C, const arma::vec &c, const arma::vec &x_minus_i,
                      bool solve)
/**
 * @brief Helps solving an LCP as an MIQP using bigM constraints. 
 * @returns A unique_ptr to GRBModel that has the equivalent MIQP
 * @details The MIQP problem that is returned by this function is equivalent to the LCP problem provided the value of bigM is large enough. The function differs from LCP::LCPasMIP by the fact that, this explicitly takes a leader objective, and returns an object with this objective. 
 * This allows quadratic leader objective. If you are aware that the leader's objective is linear, use the faster method LCP::MPECasMILP
 */
{
    auto model = this->MPECasMILP(C, c, x_minus_i, false);
    /// Note that if the matrix Q is a zero matrix, then this returns a Gurobi MILP model as opposed to MIQP model.
    /// This enables Gurobi to use its much advanced MIP solver
    if (Q.n_nonzero != 0) // If Q is zero, then just solve MIP as opposed to MIQP!
    {
        GRBLinExpr linexpr = model->getObjective(0);
        GRBQuadExpr expr{linexpr};
        for (auto it = Q.begin(); it != Q.end(); ++it)
            expr += (*it) *
                    model->getVarByName("x_" + to_string(it.row())) *
                    model->getVarByName("x_" + to_string(it.col()));
        model->setObjective(expr, GRB_MINIMIZE);
    }
    if (solve) model->optimize();
    return model;
}

void Game::LCP::write(string filename, bool append) const {
    ofstream outfile(filename, append ? ios::app : ios::out);

    outfile << nR << " rows and " << nC << " columns in the LCP\n";
    outfile << "LeadStart: " << LeadStart << " \nLeadEnd: " << LeadEnd << " \nnLeader: " << nLeader << "\n\n";

    outfile << "M: " << this->M;
    outfile << "q: " << this->q;
    outfile << "Complementarity: \n";
    for (const auto &p:this->Compl) outfile << "<" << p.first << ", " << p.second << ">" << "\t";
    outfile << "A: " << this->_A;
    outfile << "b: " << this->_b;
    outfile.close();
}
