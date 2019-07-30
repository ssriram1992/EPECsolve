#include<exception>
#include<chrono>
#include"../src/models.h"
#include"../src/games.h"
#include"../src/lcptolp.h"
#include<ctime>
#include<cstdlib>
#include<gurobi_c++.h>
#include<armadillo>
#include<iostream>
#include<random>

#define VERBOSE true

#define BOOST_TEST_MODULE EPECTest

#include<boost/test/unit_test.hpp>


using namespace std;
using namespace arma;

// BOOST_AUTO_TEST_SUITE(EPECTests)

    BOOST_AUTO_TEST_CASE(QP_Param_test) {
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
        unsigned int Nx = 2, Ny = 3, Ncons = 2;
        mat Qd(3, 3);
        Qd << 1 << 1 << -2 << endr
           << 1 << 1 << -2 << endr
           << -2 << -2 << 4 << endr;
        sp_mat Q = sp_mat(2 * Qd);
        sp_mat C(3, 2);
        C.zeros();
        C(0, 0) = 2;
        C(0, 1) = 2;
        C(2, 0) = 3;
        vec c(3);
        c << 1 << endr << -1 << endr << 1 << endr;
        sp_mat A(2, 2);
        A.zeros();
        A(1, 0) = -1;
        A(1, 1) = -1;
        mat Bd(2, 3);
        Bd << 1 << 1 << 1 << endr << -1 << 1 << -2 << endr;
        sp_mat B = sp_mat(Bd);
        vec b(2);
        b(0) = 10;
        b(1) = -1;
        /* Manual data over */

        GRBEnv env = GRBEnv();

        // Constructor
        BOOST_TEST_MESSAGE("Constructor tests");
        QP_Param q1(Q, C, A, B, c, b, &env);
        const QP_Param q_ref(q1);
        QP_Param q2(&env);
        q2.set(Q, C, A, B, c, b);
        BOOST_CHECK(q1 == q2);
        // Checking if the constructor is sensible
        BOOST_CHECK(q1.getNx() == Nx && q1.getNy() == Ny);

        // QP_Param.solve_fixed()
        BOOST_TEST_MESSAGE("QP_Param.solveFixed() test");
        arma::vec x(2);
        x(0) = -1;
        x(1) = 0.5;
        auto FixedModel = q2.solveFixed(x);
        arma::vec sol(3);
        sol << 0.5417 << endr << 5.9861 << endr << 3.4722; // Hardcoding the solution as calculated outside
        for (unsigned int i = 0; i < Ny; i++)
            BOOST_WARN_CLOSE(sol(i), FixedModel->getVar(i).get(GRB_DoubleAttr_X), 0.01);
        BOOST_CHECK_CLOSE(FixedModel->get(GRB_DoubleAttr_ObjVal), -12.757, 0.01);

        // KKT conditions for a QPC
        BOOST_TEST_MESSAGE("QP_Param.KKT() test");
        sp_mat M, N;
        vec q;
        // Hard coding the expected values for M, N and q
        mat Mhard(5, 5), Nhard(5, 2);
        vec qhard(5);
        Mhard << 2 << 2 << -4 << 1 << -1 << endr
              << 2 << 2 << -4 << 1 << 1 << endr
              << -4 << -4 << 8 << 1 << -2 << endr
              << -1 << -1 << -1 << 0 << 0 << endr
              << 1 << -1 << 2 << 0 << 0 << endr;
        Nhard << 2 << 2 << endr
              << 0 << 0 << endr
              << 3 << 0 << endr
              << 0 << 0 << endr
              << 1 << 1 << endr;
        qhard << 1 << -1 << 1 << 10 << -1;
        BOOST_CHECK_NO_THROW(q1.KKT(M, N, q)); // Should not throw any exception!
        // Following are hard requirements, if this fails, then addDummy test following this is not sensible
        BOOST_REQUIRE(Game::isZero(mat(M) - Mhard));
        BOOST_REQUIRE(Game::isZero(mat(N) - Nhard));
        BOOST_REQUIRE(Game::isZero(mat(q) - qhard));

        // addDummy
        BOOST_TEST_MESSAGE("QP_Param.addDummy(0, 1, 1) test");
        // First adding a dummy variable using QP_Param::addDummy(var, param, pos);
        q1.addDummy(0, 1, 1);
        // Position should not matter for variable addition
        Game::QP_Param q3 = QP_Param(q_ref);
        q3.addDummy(0, 1, 0);
        BOOST_CHECK_MESSAGE(q1 == q3, "Checking location should not matter for variables");


        // Q should remain same on left part, and the last row and col have to be zeros
        arma::sp_mat temp_spmat1 = q1.getQ().submat(0, 0, Ny - 1, Ny - 1); // The top left part should not have changed.
        BOOST_CHECK_MESSAGE(Game::isZero(temp_spmat1 - q_ref.getQ()), "Q check after addDummy(0, 1, 1)");
        temp_spmat1 = q1.getQ().cols(Ny, Ny);
        BOOST_CHECK_MESSAGE(Game::isZero(temp_spmat1), "Q check after addDummy(0, 1, 1)");
        temp_spmat1 = q1.getQ().rows(Ny, Ny);
        BOOST_CHECK_MESSAGE(Game::isZero(temp_spmat1), "Q check after addDummy(0, 1, 1)");
        // C  should have a new zero row below
        temp_spmat1 = q1.getC().submat(0, 0, Ny - 1, Nx - 1);
        BOOST_CHECK_MESSAGE(Game::isZero(temp_spmat1 - q_ref.getC()), "C check after addDummy(0, 1, 1)");
        temp_spmat1 = q1.getC().row(Ny);
        BOOST_CHECK_MESSAGE(Game::isZero(temp_spmat1), "C check after addDummy(0, 1, 1)");

        // A should not change
        BOOST_CHECK_MESSAGE(Game::isZero(q_ref.getA() - q1.getA()), "A check after addDummy(0, 1, 1)");

        // B
        temp_spmat1 = q1.getB().submat(0, 0, Ncons - 1, Ny - 1);
        BOOST_CHECK_MESSAGE(Game::isZero(temp_spmat1 - q_ref.getB()), "B check after addDummy(0, 1, 1)");
        temp_spmat1 = q1.getB().col(Ny);
        BOOST_CHECK_MESSAGE(Game::isZero(temp_spmat1), "B check after addDummy(0, 1, 1)");

        // b
        BOOST_CHECK_MESSAGE(Game::isZero(q_ref.getb() - q1.getb()), "b check after addDummy(0, 1, 1)");

        // c
        temp_spmat1 = q1.getc().subvec(0, Ny - 1);
        BOOST_CHECK_MESSAGE(Game::isZero(temp_spmat1 - q_ref.getc()), "c check after addDummy(0, 1, 1)");
        BOOST_CHECK_MESSAGE(abs(q1.getc().at(Ny)) < 1e-4, "c check after addDummy(0, 1, 1)");


        BOOST_TEST_MESSAGE("QP_Param.addDummy(1, 0, 0) test");
        BOOST_WARN_MESSAGE(false, "Not yet implemented");


        BOOST_TEST_MESSAGE("QP_Param.addDummy(1, 0, -1) test");
        BOOST_WARN_MESSAGE(false, "Not yet implemented");


    }


    BOOST_AUTO_TEST_CASE(NashGame_test) {
        BOOST_TEST_MESSAGE("\n\n");
        BOOST_TEST_MESSAGE("Testing Game::NashGame");

        GRBEnv env;

        /** First test is to create a duopoly **/
        /* PLAYER 1:
         * 	min: 10 q1 + 0.1 q1^2 - (100 - (q1+q2)) q1 	= 1.1 q1^2 - 90 q1 + q1q2
         * 	 s.t:
         * 	 	q1 >= 0
         *
         * PLAYER 2:
         * 	min: 5 q2 + 0.2 q2^2 - (100 - (q1+q2)) q2 	= 1.2 q2^2 - 95 q2 + q2q1
         * 	 s.t:
         * 	 	q2 >= 0
         *
         * EXPECTED LCP
         * 0 \leq q1 \perp 2.2 q1 + q2 - 90 \geq 0
         * 0 \leq q2 \perp q1 + 2.4 q2 - 95 \geq 0
         * Solution: q1=28.271, q2=27.8037
         */
        arma::sp_mat Q(1, 1), A(0, 1), B(0, 1), C(1, 1);
        arma::vec b, c(1);
        b.set_size(0);
        Q(0, 0) = 2 * 1.1;
        C(0, 0) = 1;
        c(0) = -90;
        auto q1 = std::make_shared<Game::QP_Param>(Q, C, A, B, c, b, &env);
        Q(0, 0) = 2 * 1.2;
        c(0) = -95;
        auto q2 = std::make_shared<Game::QP_Param>(Q, C, A, B, c, b, &env);

        // Creating the Nashgame
        std::vector<shared_ptr<Game::QP_Param>> q{q1, q2};
        sp_mat MC(0, 2);
        vec MCRHS;
        MCRHS.set_size(0);
        Game::NashGame Nash = Game::NashGame(q, MC, MCRHS);

        // Master check  -  LCP should be proper!
        sp_mat MM, MM_ref;
        vec qq, qq_ref;
        perps Compl;
        BOOST_TEST_MESSAGE("NashGame.FormulateLCP test");
        BOOST_CHECK_NO_THROW(Nash.FormulateLCP(MM, qq, Compl));
        BOOST_CHECK_MESSAGE(MM(0, 0) == 2.2, "checking q1 coefficient in M-LCP (0,0)");
        BOOST_CHECK_MESSAGE(MM(0, 1) == 1, "checking q2 coefficient in M-LCP (0,1)");
        BOOST_CHECK_MESSAGE(MM(1, 0) == 1, "checking q1 coefficient in M-LCP (1,0)");
        BOOST_CHECK_MESSAGE(MM(1, 1) == 2.4, "checking q2 coefficient in M-LCP (1,1)");
        BOOST_CHECK_MESSAGE(qq(0) == -90, "checking rhs coefficient in Q-LCP (0)");
        BOOST_CHECK_MESSAGE(qq(1) == -95, "checking rhs coefficient in Q-LCP (1)");

        BOOST_TEST_MESSAGE("LCP.LCPasMIP test");
        Game::LCP lcp = Game::LCP(&env, Nash);
        unique_ptr<GRBModel> lcpmodel = lcp.LCPasMIP(true);

        // int Nvar = Nash.getNprimals() + Nash.getNduals() + Nash.getNshadow() + Nash.getNleaderVars();
        BOOST_CHECK_NO_THROW(lcpmodel->getVarByName("x_0").get(GRB_DoubleAttr_X));
        BOOST_CHECK_NO_THROW(lcpmodel->getVarByName("x_1").get(GRB_DoubleAttr_X));
        BOOST_CHECK_CLOSE(lcpmodel->getVarByName("x_0").get(GRB_DoubleAttr_X), 28.271, 0.001);
        BOOST_CHECK_CLOSE(lcpmodel->getVarByName("x_1").get(GRB_DoubleAttr_X), 27.8037, 0.001);

    }


    BOOST_AUTO_TEST_CASE(ConvexHulltest) {
        BOOST_TEST_MESSAGE("\n\n");
        BOOST_TEST_MESSAGE("Testing Game::ConvexHull");

        GRBEnv env;
        arma::sp_mat A1, A2, A3, A;
        arma::vec b1, b2, b3, b;
        vector<arma::sp_mat *> Ai;
        vector<arma::vec *> bi;

        A.zeros();
        b.zeros();

        // convention A<=b
        //------FIRST POLYHEDRON
        A1.zeros(4, 2);
        b1.zeros(4);
        //x1>=0
        A1(0, 0) = -1;
        //x1<=1
        A1(1, 0) = 1;
        b1(1) = 1;
        //x2>=0
        A1(2, 1) = -1;
        //x2<=1
        A1(3, 1) = 1;
        b1(3) = 1;
        Ai.push_back(&A1);
        bi.push_back(&b1);

        //------SECOND POLYHEDRON
        A2.zeros(4, 2);
        b2.zeros(4);
        //x1<=3
        A2(0, 0) = 1;
        b2(0) = 3;
        //x1>=2
        A2(1, 0) = -1;
        b2(1) = -2;
        //x2<=1
        A2(2, 1) = 1;
        b2(2) = 1;
        //x2>=0
        A2(3, 1) = -1;
        Ai.push_back(&A2);
        bi.push_back(&b2);

		
        //------THIRD POLYHEDRON
        A3.zeros(4, 2);
        b3.zeros(4);
        //x1>=1
        A3(0, 0) = -1;
        b3(0) = -1;
        //x1<=2
        A3(1, 0) = 1;
        b3(1) = 2;
        //x2>=1
        A3(2, 1) = -1;
        b3(2) = -1;
        //x2<=1.5
        A3(3, 1) = 1;
        b3(3) = 1.5;
        Ai.push_back(&A3);
        bi.push_back(&b3);

        //Minimize the sum of negative variables. Solution should be a vertex of polyhedron A2

        GRBModel model = GRBModel(env);
        Game::ConvexHull(&Ai, &bi, A, b);
        GRBVar x[A.n_cols];
        GRBConstr a[A.n_rows];
        for (unsigned int i = 0; i < A.n_cols; i++)
            x[i] = model.addVar(-GRB_INFINITY, +GRB_INFINITY, 0, GRB_CONTINUOUS, "x_" + to_string(i));
        for (unsigned int i = 0; i < A.n_rows; i++) {
            GRBLinExpr lin{0};
            for (auto j = A.begin_row(i); j != A.end_row(i); ++j)
                lin += (*j) * x[j.col()];
            a[i] = model.addConstr(lin, GRB_LESS_EQUAL, b.at(i));
        }
        GRBLinExpr obj = 0;
        obj += x[0] + x[1];
        model.setObjective(obj, GRB_MAXIMIZE);
        model.set(GRB_IntParam_OutputFlag, 1);
        model.set(GRB_IntParam_DualReductions, 0);
        model.write("dat/ConvexHullTest.lp");
        model.optimize();
        model.write("dat/ConvexHullTest.sol");
        BOOST_CHECK_MESSAGE(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL, "checking optimization status");
        BOOST_CHECK_MESSAGE(model.getObjective().getValue() == 4,"checking obj==4");
        BOOST_CHECK_MESSAGE(model.getVarByName("x_0").get(GRB_DoubleAttr_X) == 3,"checking x0==3");
        BOOST_CHECK_MESSAGE(model.getVarByName("x_1").get(GRB_DoubleAttr_X) == 1,"checking x1==1");

    }

    BOOST_AUTO_TEST_CASE(SingleBilevelNonConvex) {
        BOOST_TEST_MESSAGE("Testing a single bilevel problem without convexification.");
        Models::FollPar FP;
        FP.capacities = {100};
        FP.costs_lin = {10};
        FP.costs_quad = {5};
        FP.emission_costs = {6};
        FP.names = {"NiceFollower"};
        Models::LeadAllPar Country(1, "NiceCountry", FP, {300, 0.05}, {290, -1, -1, -1});
        GRBEnv env = GRBEnv();
        arma::sp_mat M;
        arma::vec q;
        perps Compl;
        arma::sp_mat Aa;
        arma::vec b;
        Models::EPEC epec(&env);
        arma::sp_mat TrCo(1, 1);
        TrCo(0, 0) = 0;
        //Switch off convexification
        epec.convexify = false;

        BOOST_CHECK_NO_THROW(epec.addCountry(Country));
        BOOST_CHECK_NO_THROW(epec.addTranspCosts(TrCo));
        BOOST_CHECK_NO_THROW(epec.finalize());
        BOOST_CHECK_NO_THROW(epec.make_country_QP());
        BOOST_CHECK_NO_THROW(epec.findNashEq(true));
        BOOST_CHECK_MESSAGE(epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 0) == 0,
                            "checking q1==0");
        BOOST_CHECK_MESSAGE(epec.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 0) == 290, "checking t1==290");
    }

    BOOST_AUTO_TEST_CASE(SingleBilevel) {
        BOOST_TEST_MESSAGE("Testing a single bilevel problem without convexification.");
        Models::FollPar FP;
        FP.capacities = {100};
        FP.costs_lin = {10};
        FP.costs_quad = {5};
        FP.emission_costs = {6};
        FP.names = {"NiceFollower"};
        Models::LeadAllPar Country(1, "NiceCountry", FP, {300, 0.05}, {290, -1, -1, -1});
        GRBEnv env = GRBEnv();
        arma::sp_mat M;
        arma::vec q;
        perps Compl;
        arma::sp_mat Aa;
        arma::vec b;
        Models::EPEC epec(&env);
        arma::sp_mat TrCo(1, 1);
        TrCo(0, 0) = 0;
        //Switch off convexification
        epec.convexify = false;

        BOOST_CHECK_NO_THROW(epec.addCountry(Country));
        BOOST_CHECK_NO_THROW(epec.addTranspCosts(TrCo));
        BOOST_CHECK_NO_THROW(epec.finalize());
        BOOST_CHECK_NO_THROW(epec.make_country_QP());
        BOOST_CHECK_NO_THROW(epec.findNashEq(true));
        double q1 = epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 0), t1 = epec.x.at(
                epec.getPosition(0, Models::LeaderVars::Tax) + 0);
        BOOST_TEST_MESSAGE("Testing non-convexified results");
        BOOST_CHECK_MESSAGE(q1 == 0, "(NC) checking q1==0");
        BOOST_CHECK_MESSAGE(t1 == 290, "(NC) checking t1==290");

        Models::EPEC epec2(&env);
        BOOST_CHECK_NO_THROW(epec2.addCountry(Country));
        BOOST_CHECK_NO_THROW(epec2.addTranspCosts(TrCo));
        BOOST_CHECK_NO_THROW(epec2.finalize());
        BOOST_CHECK_NO_THROW(epec2.make_country_QP());
        BOOST_CHECK_NO_THROW(epec2.findNashEq(true));
        BOOST_TEST_MESSAGE("Testing convexified results");
        BOOST_CHECK_MESSAGE(epec2.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 0) == 0,
                            "(C) checking q1==0");
        BOOST_CHECK_MESSAGE(epec2.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 0) == 290,
                            "(C) checking t1==290");

        BOOST_TEST_MESSAGE("Testing discrepancy between the 2");
        BOOST_CHECK_MESSAGE(epec2.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 0) == q1,
                            "comparing q1 among the two");
        BOOST_CHECK_MESSAGE(epec2.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 0) == t1,
                            "comparing t1 among the two");
    }

// BOOST_AUTO_TEST_SUITE_END()
