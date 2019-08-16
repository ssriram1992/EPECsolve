#ifndef GAMES_H
#define GAMES_H

// #include"epecsolve.h"
#include "lcptolp.h"
#include <iostream>
#include <memory>
#include <gurobi_c++.h>
#include <armadillo>


using namespace std;
using namespace Game;

namespace Game {
    bool isZero(arma::mat M, double tol = 1e-6);

    bool isZero(arma::sp_mat M, double tol = 1e-6);

// bool isZero(arma::vec M, double tol = 1e-6);
///@brief struct to handle the objective params of MP_Param/QP_Param
///@details Refer QP_Param class for what Q, C and c mean.
    typedef struct QP_objective {
        arma::sp_mat Q, C;
        arma::vec c;
    } QP_objective;
///@brief struct to handle the constraint params of MP_Param/QP_Param
///@details Refer QP_Param class for what A, B and b mean.
    typedef struct QP_constraints {
        arma::sp_mat A, B;
        arma::vec b;
    } QP_constraints;

///@brief class to handle parameterized mathematical programs(MP)
    class MP_Param {
    protected:
        // Data representing the parameterized QP
        arma::sp_mat Q, A, B, C;
        arma::vec c, b;

        // Object for sizes and integrity check
        unsigned int Nx, Ny, Ncons;

        unsigned int size();

        bool dataCheck(bool forcesymm = true) const;

    public:
        // Default constructors
        MP_Param() = default;

        MP_Param(const MP_Param &M) = default;

        // Getters and setters
        virtual inline arma::sp_mat
        getQ() const final { return this->Q; }   ///< Read-only access to the private variable Q
        virtual inline arma::sp_mat
        getC() const final { return this->C; }   ///< Read-only access to the private variable C
        virtual inline arma::sp_mat
        getA() const final { return this->A; }   ///< Read-only access to the private variable A
        virtual inline arma::sp_mat
        getB() const final { return this->B; }   ///< Read-only access to the private variable B
        virtual inline arma::vec
        getc() const final { return this->c; }         ///< Read-only access to the private variable c
        virtual inline arma::vec
        getb() const final { return this->b; }         ///< Read-only access to the private variable b
        virtual inline unsigned int
        getNx() const final { return this->Nx; } ///< Read-only access to the private variable Nx
        virtual inline unsigned int
        getNy() const final { return this->Ny; } ///< Read-only access to the private variable Ny

        virtual inline MP_Param &setQ(const arma::sp_mat &Q) final {
            this->Q = Q;
            return *this;
        } ///< Set the private variable Q
        virtual inline MP_Param &setC(const arma::sp_mat &C) final {
            this->C = C;
            return *this;
        } ///< Set the private variable C
        virtual inline MP_Param &setA(const arma::sp_mat &A) final {
            this->A = A;
            return *this;
        } ///< Set the private variable A
        virtual inline MP_Param &setB(const arma::sp_mat &B) final {
            this->B = B;
            return *this;
        } ///< Set the private variable B
        virtual inline MP_Param &setc(const arma::vec &c) final {
            this->c = c;
            return *this;
        } ///< Set the private variable c
        virtual inline MP_Param &setb(const arma::vec &b) final {
            this->b = b;
            return *this;
        } ///< Set the private variable b

        virtual inline bool finalize() {
            this->size();
            return this->dataCheck();
        } ///< Finalize the MP_Param object.

        // Setters and advanced constructors
        virtual MP_Param &set(const arma::sp_mat &Q, const arma::sp_mat &C,
                              const arma::sp_mat &A, const arma::sp_mat &B, const arma::vec &c,
                              const arma::vec &b); // Copy data into this
        virtual MP_Param &set(arma::sp_mat &&Q, arma::sp_mat &&C,
                              arma::sp_mat &&A, arma::sp_mat &&B, arma::vec &&c, arma::vec &&b); // Move data into this
        virtual MP_Param &set(const QP_objective &obj, const QP_constraints &cons);

        virtual MP_Param &set(QP_objective &&obj, QP_constraints &&cons);

        virtual MP_Param &addDummy(unsigned int pars, unsigned int vars = 0, int position = -1);

        void write(string filename, bool append = true) const;

        static bool
        dataCheck(const QP_objective &obj, const QP_constraints &cons, bool checkObj = true, bool checkCons = true);
    };

///@brief Class to handle parameterized quadratic programs(QP)
    class QP_Param : public MP_Param
// Shape of C is Ny\times Nx
/**
 * Represents a Parameterized QP as \f[
 * \min_y \frac{1}{2}y^TQy + c^Ty + (Cx)^T y
 * \f]
 * Subject to
 * \f{eqnarray}{
 * Ax + By &\leq& b \\
 * y &\geq& 0
 * \f}
*/
    {
    private:
        // Gurobi environment and model
        GRBEnv *env;
        GRBModel QuadModel;
        bool made_yQy;

        int make_yQy();

    public: // Constructors
        /// Initialize only the size. Everything else is empty (can be updated later)
        QP_Param(GRBEnv *env = nullptr) : env{env}, QuadModel{(*env)}, made_yQy{false} { this->size(); }

        /// Set data at construct time
        QP_Param(arma::sp_mat Q, arma::sp_mat C, arma::sp_mat A, arma::sp_mat B, arma::vec c, arma::vec b,
                 GRBEnv *env = nullptr) : env{env}, QuadModel{(*env)}, made_yQy{false} {
            this->set(Q, C, A, B, c, b);
            this->size();
            if (!this->dataCheck())
                throw string("Error in QP_Param::QP_Param: Invalid data for constructor");
        }

        /// Copy constructor
        QP_Param(const QP_Param &Qu) : MP_Param(Qu),
                                       env{Qu.env}, QuadModel{Qu.QuadModel}, made_yQy{Qu.made_yQy} { this->size(); };


        // Override setters
        QP_Param &set(const arma::sp_mat &Q, const arma::sp_mat &C,
                      const arma::sp_mat &A, const arma::sp_mat &B, const arma::vec &c,
                      const arma::vec &b) final; // Copy data into this
        QP_Param &set(arma::sp_mat &&Q, arma::sp_mat &&C,
                      arma::sp_mat &&A, arma::sp_mat &&B, arma::vec &&c, arma::vec &&b) final; // Move data into this
        QP_Param &set(const QP_objective &obj, const QP_constraints &cons) final;

        QP_Param &set(QP_objective &&obj, QP_constraints &&cons) final;

        bool operator==(const QP_Param &Q2) const;

        // Other methods
        unsigned int KKT(arma::sp_mat &M, arma::sp_mat &N, arma::vec &q) const;

        std::unique_ptr<GRBModel> solveFixed(arma::vec x);

        inline bool is_Playable(const QP_Param &P) const
        /// Checks if the current object can play a game with another Game::QP_Param object @p P.
        {
            bool b1, b2, b3;
            b1 = (this->Nx + this->Ny) == (P.getNx() + P.getNy());
            b2 = this->Nx >= P.getNy();
            b3 = this->Ny <= P.getNx();
            return b1 && b2 && b3;
        }

        QP_Param &addDummy(unsigned int pars, unsigned int vars = 0, int position = -1) override;

        void write(string filename, bool append) const;
		void save(string filename, bool erase = true) const;
		long int load(string filename, long int pos = 0);
    };

/**
 * @brief Class to model Nash-cournot games with each player playing a QP
 */
/**
 * Stores a vector of QPs with each player's optimization problem.
 * Potentially common (leader) constraints can be stored too.
 *
 * Helpful in rewriting the Nash-Cournot game as an LCP
 * Helpful in rewriting leader constraints after incorporating dual variables etc
 * @warning This has public fields which if accessed and changed can cause
 * undefined behavior! 
 * \todo Better implementation which will make the above warning go away!
 */
    class NashGame {
    private:
		GRBEnv* env=nullptr;
        arma::sp_mat LeaderConstraints;          ///< Upper level leader constraints LHS
        arma::vec LeaderConsRHS;              ///< Upper level leader constraints RHS
        unsigned int Nplayers;                  ///< Number of players in the Nash Game
        vector<shared_ptr<QP_Param>> Players; ///< The QP that each player solves
        arma::sp_mat MarketClearing;          ///< Market clearing constraints
        arma::vec MCRHS;                      ///< RHS to the Market Clearing constraints

        /// @internal In the vector of variables of all players,
        /// which position does the variable corrresponding to this player starts.
        vector<unsigned int> primal_position;
        ///@internal In the vector of variables of all players,
        /// which position do the DUAL variable corrresponding to this player starts.
        vector<unsigned int> dual_position;
        /// @internal Manages the position of Market clearing constraints' duals
        unsigned int MC_dual_position;
        /// @internal Manages the position of where the leader's variables start
        unsigned int Leader_position;
        /// Number of leader variables.
        /// These many variables will not have a matching complementary equation.
        unsigned int n_LeadVar;

        void set_positions();

    public: // Constructors
		NashGame(GRBEnv* e):env{e}{}; ///< To be used only when NashGame is being loaded from a file.
        NashGame(vector<shared_ptr<QP_Param>> Players, arma::sp_mat MC,
                 arma::vec MCRHS, unsigned int n_LeadVar = 0, arma::sp_mat LeadA = {}, arma::vec LeadRHS = {});

        NashGame(unsigned int Nplayers, unsigned int n_LeadVar = 0, arma::sp_mat LeadA = {}, arma::vec LeadRHS = {})
                : LeaderConstraints{LeadA}, LeaderConsRHS{LeadRHS}, Nplayers{Nplayers}, n_LeadVar{n_LeadVar} {
            Players.resize(this->Nplayers);
            primal_position.resize(this->Nplayers);
            dual_position.resize(this->Nplayers);
        }

        /// Destructors to `delete` the QP_Param objects that might have been used.
        ~NashGame() {};

        // Verbose declaration
        friend ostream &operator<<(ostream &os, const NashGame &N) {
            os << endl;
            os << "-----------------------------------------------------------------------" << endl;
            os << "Nash Game with " << N.Nplayers << " players" << endl;
            os << "-----------------------------------------------------------------------" << endl;
            os << "Number of primal variables:\t\t\t " << N.getNprimals() << endl;
            os << "Number of dual variables:\t\t\t " << N.getNduals() << endl;
            os << "Number of shadow price dual variables:\t\t " << N.getNshadow() << endl;
            os << "Number of leader variables:\t\t\t " << N.getNleaderVars() << endl;
            os << "-----------------------------------------------------------------------" << endl;
            return os;
        }

        /// Return the number of primal variables
        inline unsigned int getNprimals() const { return this->primal_position.back(); }

        inline unsigned int getNshadow() const { return this->MCRHS.n_rows; }

        inline unsigned int getNleaderVars() const { return this->n_LeadVar; }

        inline unsigned int getNduals() const { return this->dual_position.back() - this->dual_position.front() + 0; }

        // Size of variables
        inline unsigned int getPrimalLoc(unsigned int i = 0) const { return primal_position.at(i); }

        inline unsigned int getMCdualLoc() const { return MC_dual_position; }

        inline unsigned int getLeaderLoc() const { return Leader_position; }

        inline unsigned int getDualLoc(unsigned int i = 0) const { return dual_position.at(i); }

        // Members
        const NashGame &
        FormulateLCP(arma::sp_mat &M, arma::vec &q, perps &Compl, bool writeToFile = false,
                     string M_name = "dat/LCP.txt",
                     string q_name = "dat/q.txt") const;

        arma::sp_mat RewriteLeadCons() const;

        inline arma::vec getLeadRHS() const { return this->LeaderConsRHS; }

        inline arma::vec getMCLeadRHS() const {
            return arma::join_cols(
                    arma::join_cols(this->LeaderConsRHS, this->MCRHS),
                    -this->MCRHS);
        }

        NashGame &addDummy(unsigned int par = 0, int position = -1);

        NashGame &addLeadCons(const arma::vec &a, double b);

        void write(string filename, bool append = true, bool KKT = false) const;
		void save(string filename, bool erase=true) const;
		long int load(string filename, long int pos = 0); 
    };

// void MPEC(NashGame N, arma::sp_mat Q, QP_Param &P);
// ostream& operator<< (ostream& os, const QP_Param &Q);
// void MPEC(NashGame N, arma::sp_mat Q, QP_Param &P);
    ostream &operator<<(ostream &os, const QP_Param &Q);

    ostream &operator<<(ostream &ost, const perps &C);

    void print(const perps &C);
}; // namespace Game

#endif
