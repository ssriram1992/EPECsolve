#include<exception>
#include<chrono>
#include"../src/models.h"
#include"../src/games.h"
#include<ctime>
#include<cstdlib>
#include<gurobi_c++.h>
#include<armadillo>
#include<iostream>
#include<random>

#define BOOST_TEST_MODULE EPECTest
// #include<boost/test/included/unit_test.hpp>
#include<boost/test/unit_test.hpp>

#define TRY_BLOCK(x) try{x;}catch(const string &e){cerr<<e<<endl;}

using namespace std;
using namespace arma;

/* Random number generation routines */
// This for more realistic randomness
// std::default_random_engine g(std::chrono::system_clock::now().time_since_epoch().count());
// The below for repeatability
std::default_random_engine g(4);
std::uniform_int_distribution<int> rand_binary(0,1);
std::uniform_real_distribution<double> rand_01_real(0,1);
std::uniform_int_distribution<int> rand_int(0, 5);



BOOST_AUTO_TEST_CASE(QP_Param_test)
        {
                BOOST_TEST_MESSAGE("\n\n");
        BOOST_TEST_MESSAGE("Testing Game::QP_Param");

        /* Random data
        arma_rng::set_seed(rand_int(g));
        unsigned int Ny = 2+rand_int(g);
        unsigned int Nx = 1+rand_int(g);
        unsigned int Nconstr = (rand_int(g)+2)/2;
        sp_mat Q = arma::sprandu<sp_mat>(Ny, Ny, 0.2);
        Q = Q*Q.t();
        BOOST_REQUIRE(Q.is_symmetric());
        Q.print_dense();
        sp_mat C = arma::sprandu<sp_mat>(Ny, Nx, 0.3);
        sp_mat A = arma::sprandu<sp_mat>(Nconstr, Nx, 0.7);
        sp_mat B = arma::sprandu<sp_mat>(Nconstr, Ny, 0.7);
        vec b(Nconstr); for (unsigned int i = 0; i<Nconstr; ++i) b(i) = 3*rand_int(g);
        vec c = arma::randg(Ny);
        */

        /* Below is the data for the following quadratic programming problem
         * min (y1 + y2 - 2y3)^2 + 2 x1y1 + 2 x2y1 + 3 x1y3 + y1-y2+y3
         * Subject to
         * y1, y2, y3 >= 0
         * y1 + y2 + y3 <= 10
         * -y1 +y2 -2y3 <= -1 + x1 + x2
         *
         * With (x1, x2) = (-1, 0.5), problem is
         * min (y1 + y2 - 2y3)^2  -y2 -2y3
         * Subject to
         * y1, y2, y3 >= 0
         * y1 + y2 + y3 <= 10
         * -y1 +y2 -2y3 <= -1.5
         *
         *  The optimal objective value for this problem (as solved outside) is -12.757
         *  and a potential solution (y1, y2, y3) is (0.542, 5.986, 3.472)
         *
         */
        unsigned int Nx = 2, Ny = 3;
        mat Qd(3,3);
        Qd<<1<<1<<-2<<endr
        <<1<<1<<-2<<endr
        <<-2<<-2<<4<<endr;
        sp_mat Q = sp_mat(2*Qd);
        sp_mat C(3, 2); C.zeros();
        C(0,0) = 2; C(0,1) = 2; C(2, 0) = 3;
        vec c(3); c<<1<<endr<<-1<<endr<<1<<endr;
        sp_mat A(2, 2); A.zeros();
        A(1,0) = -1; A(1,1) = -1;
        mat Bd(2, 3);
        Bd<<1<<1<<1<<endr<<-1<<1<<-2<<endr;
        sp_mat B = sp_mat(Bd);
        vec b(2); b(0) = 10; b(1) = -1;
        /* Manual data over */

        GRBEnv env = GRBEnv();

        // Constructor
        BOOST_TEST_MESSAGE("Constructor tests");
        QP_Param q1(Q, C, A, B, c, b, &env);
        QP_Param q2(&env);
        q2.set(Q, C, A, B, c, b);
        BOOST_CHECK(q1 == q2 );
        // Checking if the constructor is sensible
        BOOST_CHECK(q1.getNx() == Nx && q1.getNy() == Ny);

        // QP_Param.solve_fixed()
        BOOST_TEST_MESSAGE("QP_Param.solveFixed() test");
        arma::vec x(2); x(0) = -1; x(1) = 0.5;
        auto FixedModel = q2.solveFixed(x);
        arma::vec sol(3); sol<<0.5417<<endr<<5.9861<<endr<<3.4722; // Hardcoding the solution as calculated outside
        for(unsigned int i=0; i<Ny; i++)
        BOOST_WARN_CLOSE(sol(i), FixedModel->getVar(i).get(GRB_DoubleAttr_X), 0.01);
        BOOST_CHECK_CLOSE(FixedModel->get(GRB_DoubleAttr_ObjVal), -12.757 ,0.01 );

        // KKT conditions for a QP
        BOOST_TEST_MESSAGE("QP_Param.KKT() test");
        sp_mat M, N; vec q;
        // Hard coding the expected values for M, N and q
        mat Mhard(5,5), Nhard(5,2); vec qhard(5);
        Mhard<<2<<2<<-4<<1<<-1<<endr
        <<2<<2<<-4<<1<<1<<endr
        <<-4<<-4<<8<<1<<-2<<endr
        <<-1<<-1<<-1<<0<<0<<endr
        <<1<<-1<<2<<0<<0<<endr;
        Nhard <<2<<2<<endr
        <<0<<0<<endr
        <<3<<0<<endr
        <<0<<0<<endr
        <<1<<1<<endr;
        qhard<<1<<-1<<1<<10<<-1;
        BOOST_CHECK_NO_THROW(q1.KKT(M, N, q)); // Should not throw any exception!
        // Following are hard requirements, if this fails, then addDummy test following this is not sensible
        BOOST_REQUIRE(Game::isZero(mat(M)- Mhard));
        BOOST_REQUIRE(Game::isZero(mat(N)- Nhard));
        BOOST_REQUIRE(Game::isZero(mat(q)- qhard));

        // addDummy
        BOOST_TEST_MESSAGE("QP_Param.addDummy() test");
        // First adding a dummy variable
        q1.addDummy(0, 1, 1);
        q1.KKT(M,N,q);
        M.print_dense("Mnew"); N.print_dense("Nnew"); q.print("qnew");
        q1.getC().print_dense("changed");
        q2.getC().print_dense("orig");


                // delete env;

        }





/* These are tests for Models.h and using EPEC class.
 * We make increasingly complicated problems and test them.
 * Bella ciao */

BOOST_AUTO_TEST_CASE(SingleBilevel)
        {
                BOOST_TEST_MESSAGE("\n\n");
        BOOST_WARN_MESSAGE(false, "Not yet implemented");
        }