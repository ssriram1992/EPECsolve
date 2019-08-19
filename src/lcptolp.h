#ifndef LCPTOLP_H
#define LCPTOLP_H

#include "epecsolve.h"
#include <iostream>
#include <memory>
#include <gurobi_c++.h>
#include <armadillo>

//using namespace Game;

namespace Game {

    arma::vec
    LPSolve(const arma::sp_mat &A, const arma::vec &b, const arma::vec &c, int &status, bool Positivity = false);

    int ConvexHull(const vector<arma::sp_mat *> *Ai, const vector<arma::vec *> *bi, arma::sp_mat &A, arma::vec &b,
                   const arma::sp_mat Acom = {}, const arma::vec bcom = {});

    void compConvSize(arma::sp_mat &A, const unsigned int nFinCons, const unsigned int nFinVar,
                      const vector<arma::sp_mat *> *Ai, const vector<arma::vec *> *bi,
                      const arma::sp_mat &Acom, const arma::vec &bcom);
/**
 * @brief Class to handle and solve linear complementarity problems
 */
/**
* A class to handle linear complementarity problems (LCP)
* especially as MIPs with bigM constraints
* Also provides the convex hull of the feasible space, restricted feasible space etc.
*/
    class LCP {
    private:
        // Essential data ironment for MIP/LP solves
        GRBEnv *env;    ///< Gurobi env
        arma::sp_mat M; ///< M in @f$Mx+q@f$ that defines the LCP
        arma::vec q;    ///< q in @f$Mx+q@f$ that defines the LCP
        perps Compl;    ///< Compl stores data in <Eqn, Var> form.
        unsigned int LeadStart{1}, LeadEnd{0}, nLeader{0};
        arma::sp_mat _A = {};
        arma::vec _b = {};           ///< Apart from @f$0 \le x \perp Mx+q\ge 0@f$, one needs@f$ Ax\le b@f$ too!
        // Temporary data
        bool madeRlxdModel{false}; ///< Keep track if LCP::RlxdModel is made
        unsigned int nR, nC;
        int polyCounter{0};
        int feasiblePolyhedra{-1};
        /// LCP feasible region is a union of polyhedra. Keeps track which of those inequalities are fixed to equality to get the individual polyhedra
        vector<vector<short int> *> *AllPolyhedra, *RelAllPol;
        vector<arma::sp_mat *> *Ai, *Rel_Ai;
        vector<arma::vec *> *bi, *Rel_bi;
        GRBModel RlxdModel; ///< A gurobi model with all complementarity constraints removed.

        bool errorCheck(bool throwErr = true) const;

        void defConst(GRBEnv *env);

        void makeRelaxed();

        /* Solving relaxations and restrictions */
        unique_ptr<GRBModel> LCPasMIP(vector<unsigned int> FixEq = {},
                                      vector<unsigned int> FixVar = {}, bool solve = false);

        unique_ptr<GRBModel> LCPasMIP(vector<short int> Fixes, bool solve);

        unique_ptr<GRBModel> LCP_Polyhed_fixed(vector<unsigned int> FixEq = {},
                                               vector<unsigned int> FixVar = {});

        unique_ptr<GRBModel> LCP_Polyhed_fixed(arma::Col<int> FixEq,
                                               arma::Col<int> FixVar);

        /* Branch and Prune Methods */
        template<class T>
        inline bool isZero(const T val) const { return (val > -eps && val < eps); }

        inline vector<short int> *solEncode(GRBModel *model) const;

        vector<short int> *solEncode(const arma::vec &z, const arma::vec &x) const;

        void branch(int loc, const vector<short int> *Fixes);

        vector<short int> *anyBranch(const vector<vector<short int> *> *vecOfFixes, vector<short int> *Fix) const;

        int branchLoc(unique_ptr<GRBModel> &m, vector<short int> *Fix);

        int branchProcLoc(vector<short int> *Fix, vector<short int> *Leaf);

        LCP &EnumerateAll(bool solveLP = false);

        LCP &FixToPoly(const vector<short int> *Fix, bool checkFeas = false, bool custom = false,
                       vector<arma::sp_mat *> *custAi = {}, vector<arma::vec *> *custbi = {});

        LCP &FixToPolies(const vector<short int> *Fix, bool checkFeas = false, bool custom = false,
                         vector<arma::sp_mat *> *custAi = {}, vector<arma::vec *> *custbi = {});

    public:
        // Fudgible data
        long double bigM{1e7}; ///< bigM used to rewrite the LCP as MIP
        long double eps{1e-5}; ///< The threshold for optimality and feasability tollerances
        long double eps_int{1e-8}; ///< The threshold, below which a number would be considered to be zero.
        bool useIndicators{
                true};///< If true, complementarities will be handled with indicator constraints. BigM formulation otherwise

        /** Constructors */
        /// Class has no default constructors
        LCP() = delete;

        LCP(GRBEnv *e) : env{e}, RlxdModel(*e) {}; ///< This constructor flor loading LCP from a file

        LCP(GRBEnv *env, arma::sp_mat M, arma::vec q,
            unsigned int LeadStart, unsigned LeadEnd, arma::sp_mat A = {},
            arma::vec b = {}); // Constructor with M,q,leader posn
        LCP(GRBEnv *env, arma::sp_mat M, arma::vec q,
            perps Compl, arma::sp_mat A = {}, arma::vec b = {}); // Constructor with M, q, compl pairs
        LCP(GRBEnv *env, const Game::NashGame &N);

        /** Destructor - to delete the objects created with new operator */
        ~LCP();

        /** Return data and address */
        inline arma::sp_mat getM() { return this->M; }           ///< Read-only access to LCP::M
        inline arma::sp_mat *getMstar() { return &(this->M); } ///< Reference access to LCP::M
        inline arma::vec getq() { return this->q; }               ///< Read-only access to LCP::q
        inline arma::vec *getqstar() { return &(this->q); }    ///< Reference access to LCP::q
        inline unsigned int getLStart() { return LeadStart; }  ///< Read-only access to LCP::LeadStart
        inline unsigned int getLEnd() { return LeadEnd; }      ///< Read-only access to LCP::LeadEnd
        inline perps getCompl() { return this->Compl; }           ///< Read-only access to LCP::Compl
        void print(string end = "\n");                           ///< Print a summary of the LCP
        inline unsigned int getNcol() { return this->M.n_cols; };

        inline unsigned int getNrow() { return this->M.n_rows; };


        bool extractSols(GRBModel *model, arma::vec &z,
                         arma::vec &x, bool extractZ = false) const;

        vector<vector<short int> *> *BranchAndPrune();

        /* Getting single point solutions */
        unique_ptr<GRBModel> LCPasQP(bool solve = false);

        unique_ptr<GRBModel> LCPasMIP(bool solve = false);

        unique_ptr<GRBModel>
        MPECasMILP(const arma::sp_mat &C, const arma::vec &c, const arma::vec &x_minus_i, bool solve = false);

        unique_ptr<GRBModel>
        MPECasMIQP(const arma::sp_mat &Q, const arma::sp_mat &C, const arma::vec &c, const arma::vec &x_minus_i,
                   bool solve = false);


        /* Convex hull computation */
        LCP &addPolyhedron(const vector<short int> &Fix, vector<arma::sp_mat *> &custAi, vector<arma::vec *> &custbi,
                           arma::sp_mat *A = {}, arma::vec *b = {});

        int ConvexHull(
                arma::sp_mat &A, ///< Convex hull inequality description LHS to be stored here
                arma::vec &b)    ///< Convex hull inequality description RHS to be stored here
        /**
             * Computes the convex hull of the feasible region of the LCP
             * @warning To be run only after LCP::BranchAndPrune is run. Otherwise this can give errors
             * @todo Formally call LCP::BranchAndPrune or throw an exception if this method is not already run
             */
        {
            return Game::ConvexHull(this->Ai, this->bi, A, b, this->_A, this->_b);
        };

        LCP &makeQP(const vector<short int> &Fix, vector<arma::sp_mat *> &custAi, vector<arma::vec *> &custbi,
                    Game::QP_objective &QP_obj, Game::QP_Param &QP);

        LCP &makeQP(Game::QP_objective &QP_obj, Game::QP_Param &QP);

        const int getFeasiblePolyhedra() const { return this->feasiblePolyhedra; }

        void write(string filename, bool append = true) const;

        void save(string filename, bool erase = true) const;

        long int load(string filename, long int pos = 0);
    };
}; // namespace Game
#endif

/* Example for LCP  */
/**
 * @page LCP_Example Game::LCP Example
 * Before reading this page, please ensure you are aware of the functionalities described in @link NashGame_Example Game::NashGame tutorial @endlink before following this page.
 *
 * Consider the Following linear complementarity problem with constraints
 * @f{eqnarray}{
 Ax + By \leq b\\
 0 \leq x \perp Mx + Ny + q \geq 0
 * @f}
 * These are the types of problems that are handled by the class Game::LCP but we use a different notation. Instead of using @p y to refer to the variables that don't have matching complementary equations, we call @i all the variables as @p x and we keep track of the position of variables which are not complementary to any equation.
 *
 * <b>Points to note: </b>
 * - The set of indices of @p x which are not complementary to any equation should be a consecutive set of indices. For consiceness, these components will be called as <i>Leader vars components</i> of @p x.
 * - Suppose the leader vars components of @p x are removed from @p x, in the remaining components, the first component should be complementary to the first row defined by @p M, second component should be complementary to the second row defined by @p M and so on. 
 *
 * Now consider the following linear complementarity problem.
 * @f{align*}{
 	x_1 + x_2 + x_3 \le 12\\
	0\le x_1 \perp x_4 - 1 \ge 0\\
	0\le x_2 \le 2 \\
	0 \le x_3 \perp 2x_3 + x_5 \ge 0\\
	0 \le x_4 \perp -x_1 + x_2 + 10 \ge 0\\
	0 \le x_5 \perp x_2 - x_3 + 5 \ge 0
 * @f}
 * Here indeed @f$ x_2 @f$ is the leader vars component with no complementarity equation. This problem can be entered into the Game::LCP class as follows.
 * @code
		arma::sp_mat M(4, 5); // We have four complementarity eqns and 5 variables.
		arma::vec q(4);
		M.zeros(); 
		// First eqn
		M(0, 3) = 1;
		q(0) = -1;
		// Second eqn
		M(1, 2) = 2;
		M(1, 4)  = 1;
		q(1) = 0;
		// Third eqn
		M(2, 0) = -1;
		M(2, 1) = 1;
		q(2) = 10;
		// Fourth eqn
		M(3, 1) = 1 ;
		M(3, 2) = -1;
		q(3) = 5;
		// Other common constraints
		arma::sp_mat A(2, 5); arma::vec b;
		A.zeros(); 
		// x_2 <= 2 constraint
		A(0, 1) = 1;
		b(0) = 2;
		// x_1 + x_2 + x_3 <= 12 constraint
		A(1, 0) = 1;
		A(1, 1) = 1;
		A(1, 2) = 1;
		b(1) = 12;
 * @endcode
 *
 * Now, since the variable with no complementarity pair is @f$x_2@f$ which is in position @p 1 (counting from 0) of the vector @p x, the arguments @p LeadStart and @LeadEnd in the constructor, Game::LCP::LCP are @p 1 as below.
 * @code
		GRBEnv env;
		LCP lcp = LCP(&env, M, q, 1, 1, A, b);
 * @endcode
 * This problem can be solved either using big-M based disjunctive formulation with the value of the @p bigM can also be chosen. But a more preferred means of solving is by using indicator constraints, where the algorithm tries to automatically identify good choices of bigM for each disjunction. Use the former option, only if you are very confident of  your choice of a small value of @p bigM.
 * @code
 // Solve using bigM constraints
 lcp.useIndicators = false;
 lcp.bigM = 1e5;
 auto bigMModel = lcp.LCPasMIP(true);

 // Solve using indicator constraints
 lcp.useIndicators = true;
 auto indModel = lcp.LCPasMIP(true);
 * @endcode
 * Both @p bigMModel and @p indModel are unique_ptr  to GRBModel objects. So all native gurobi operations can be performed on these objects.
 *
 * This LCP as multiple solutions. In fact the solution set can be parameterized as below.
 * @f{align}{
 x_1 &= 10 + t\\
 x_2 &= t\\
 x_3 &= 0\\
 x_4 &= 1\\
 x_5 &= 0
 @f}
 * for @f$t \in [0, 1]@f$.
 *
 But some times, one might want to solve an MPEC. i.e., optimize over the feasible region of the set as decribed above. For this purpose, two functions Game::LCP::MPECasMILP and Game::LCP::MPECasMIQP are available, depending upon whether one wants to optimize a linear objective function or a convex quadratic objective function over the set of solutions.
 *
 * 
 *
 */
