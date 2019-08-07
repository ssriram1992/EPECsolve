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

#define BOOST_TEST_MODULE EPECTest

#include<boost/test/unit_test.hpp>


using namespace std;
using namespace arma;

BOOST_AUTO_TEST_SUITE(Core__Tests)

    /* This test suite perform basic unit tests for core components (eg, QP_Param, NashGame, LCPs).
     * Also, indicator constraints are being tested for numerical stability purposes
     */

    BOOST_AUTO_TEST_CASE(QPParam_test) {
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

    BOOST_AUTO_TEST_CASE(ConvexHull_test) {

        /** Testing the convexHull method
         *  We pick three polyhedra in a two dimensional space and optimize a linear function (maximaze sum of two dimensions)
         * **/
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
        BOOST_TEST_MESSAGE("Testing Game::ConvexHull with a two dimensional problem.");
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
        BOOST_TEST_MESSAGE("Comparing results:");
        BOOST_CHECK_MESSAGE(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL, "checking optimization status");
        BOOST_CHECK_MESSAGE(model.getObjective().getValue() == 4, "checking obj==4");
        BOOST_CHECK_MESSAGE(model.getVarByName("x_0").get(GRB_DoubleAttr_X) == 3, "checking x0==3");
        BOOST_CHECK_MESSAGE(model.getVarByName("x_1").get(GRB_DoubleAttr_X) == 1, "checking x1==1");

    }

    BOOST_AUTO_TEST_CASE(IndicatorConstraints_test) {
        /** Testing the indicator constraints switch
        *  Two identical problems should have same solutions with bigM formulation and indicator constraints one
         * Numerical issues in some instances suggest that indicators are a safer choice for numerical stability issues.
        **/
        BOOST_TEST_MESSAGE("Indicator constraints test");
        Models::FollPar FP;
        FP.capacities = {100};
        FP.costs_lin = {10};
        FP.costs_quad = {0.5};
        FP.emission_costs = {6};
        FP.names = {"Blu"};
        Models::LeadAllPar Country(1, "One", FP, {300, 0.05}, {250, -1, -1, -1});
        BOOST_TEST_MESSAGE("MaxTax:250 with alpha=300 and beta=0.05");
        BOOST_TEST_MESSAGE("Expected: q=66.666;t=250");
        GRBEnv env = GRBEnv();
        arma::sp_mat TrCo(1, 1);
        TrCo(0, 0) = 0;

        BOOST_TEST_MESSAGE("----Testing Models with indicator constraints----");
        Models::EPEC epec2(&env);
        BOOST_TEST_MESSAGE("testing Models::addCountry");
        BOOST_CHECK_NO_THROW(epec2.addCountry(Country));
        BOOST_TEST_MESSAGE("testing Models::addTranspCost");
        BOOST_CHECK_NO_THROW(epec2.addTranspCosts(TrCo));
        BOOST_TEST_MESSAGE("testing Models::finalize");
        BOOST_CHECK_NO_THROW(epec2.finalize());
        BOOST_TEST_MESSAGE("testing Models::make_country_QP");
        BOOST_CHECK_NO_THROW(epec2.make_country_QP());
        BOOST_TEST_MESSAGE("testing Models::findNashEq");
        BOOST_CHECK_NO_THROW(epec2.findNashEq(true));
        BOOST_TEST_MESSAGE("Testing results (with indicators)");
        double q1 = epec2.x.at(epec2.getPosition(0, Models::LeaderVars::FollowerStart) + 0), t1 = epec2.x.at(
                epec2.getPosition(0, Models::LeaderVars::Tax) + 0);
        BOOST_CHECK_CLOSE(t1, 250, 0.001);
        BOOST_CHECK_CLOSE(q1, 66.6666, 0.01);

        Models::EPEC epec3(&env);
        epec3.indicators = false;
        BOOST_TEST_MESSAGE("----Testing Models with bigM constraints----");
        BOOST_TEST_MESSAGE("testing Models::addCountry");
        BOOST_CHECK_NO_THROW(epec3.addCountry(Country));
        BOOST_TEST_MESSAGE("testing Models::addTranspCost");
        BOOST_CHECK_NO_THROW(epec3.addTranspCosts(TrCo));
        BOOST_TEST_MESSAGE("testing Models::finalize");
        BOOST_CHECK_NO_THROW(epec3.finalize());
        BOOST_TEST_MESSAGE("testing Models::make_country_QP");
        BOOST_CHECK_NO_THROW(epec3.make_country_QP());
        BOOST_TEST_MESSAGE("testing Models::findNashEq");
        BOOST_CHECK_NO_THROW(epec3.findNashEq(true));
        BOOST_TEST_MESSAGE("Testing results (with bigM)");
        BOOST_CHECK_CLOSE(epec3.x.at(epec3.getPosition(0, Models::LeaderVars::FollowerStart) + 0), q1, 0.001);
        BOOST_CHECK_CLOSE(epec3.x.at(epec3.getPosition(0, Models::LeaderVars::Tax) + 0), t1, 0.001);


    }

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Models_Bilevel__Test)
    /* This test suite perform unit tests for EPEC problems with one country and one follower, namely Stackelberg games.
     */

    BOOST_AUTO_TEST_CASE(Bilevel_test) {

        /** Testing a Single country (C1) with a single follower (F1)
         *  LeaderConstraints: no leader constraints are enforced
        **/
        BOOST_TEST_MESSAGE("Testing a single bilevel problem with no leader constraints.");
        Models::FollPar FP;
        FP.capacities = {100};
        FP.costs_lin = {10};
        FP.costs_quad = {0.5};
        FP.emission_costs = {6};
        FP.names = {"Blu"};
        Models::LeadAllPar Country1(FP.capacities.size(), "One", FP, {300, 0.05}, {-1, -1, -1, -1});
        GRBEnv env = GRBEnv();
        arma::sp_mat TrCo(1, 1);
        TrCo(0, 0) = 0;

        // @todo change tax limit

        BOOST_TEST_MESSAGE("MaxTax:20 with alpha=300 and beta=0.05");
        BOOST_TEST_MESSAGE("Expected: q=0;t=290");
        Models::EPEC epec(&env);
        BOOST_TEST_MESSAGE("testing Models::addCountry");
        BOOST_CHECK_NO_THROW(epec.addCountry(Country1));
        BOOST_TEST_MESSAGE("testing Models::addTranspCost");
        BOOST_CHECK_NO_THROW(epec.addTranspCosts(TrCo));
        BOOST_TEST_MESSAGE("testing Models::finalize");
        BOOST_CHECK_NO_THROW(epec.finalize());
        BOOST_TEST_MESSAGE("testing Models::make_country_QP");
        BOOST_CHECK_NO_THROW(epec.make_country_QP());
        BOOST_TEST_MESSAGE("testing Models::findNashEq");
        BOOST_CHECK_NO_THROW(epec.findNashEq(true));
        BOOST_TEST_MESSAGE("Testing results:");
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 0), 0, 0.001);
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 0), 290, 0.01);
    }

    BOOST_AUTO_TEST_CASE(Bilevel_TaxCap_test) {

        /** Testing a Single country (C1) with a single follower (F1)
         *  LeaderConstraints: tax cap to 20
         *  The leader will maximize the tax (20) on the follower, which will produce q=100
        **/
        BOOST_TEST_MESSAGE("Testing a single bilevel problem with low taxcap.");
        Models::FollPar FP;
        FP.capacities = {100};
        FP.costs_lin = {10};
        FP.costs_quad = {0.5};
        FP.emission_costs = {6};
        FP.names = {"Blu"};
        Models::LeadAllPar Country1(FP.capacities.size(), "One", FP, {300, 0.05}, {20, -1, -1, -1});
        GRBEnv env = GRBEnv();
        arma::sp_mat TrCo(1, 1);
        TrCo(0, 0) = 0;

        // @todo change tax limit

        BOOST_TEST_MESSAGE("MaxTax:20 with alpha=300 and beta=0.05");
        BOOST_TEST_MESSAGE("Expected: q=100;t=20");
        Models::EPEC epec(&env);
        BOOST_TEST_MESSAGE("testing Models::addCountry");
        BOOST_CHECK_NO_THROW(epec.addCountry(Country1));
        BOOST_TEST_MESSAGE("testing Models::addTranspCost");
        BOOST_CHECK_NO_THROW(epec.addTranspCosts(TrCo));
        BOOST_TEST_MESSAGE("testing Models::finalize");
        BOOST_CHECK_NO_THROW(epec.finalize());
        BOOST_TEST_MESSAGE("testing Models::make_country_QP");
        BOOST_CHECK_NO_THROW(epec.make_country_QP());
        BOOST_TEST_MESSAGE("testing Models::findNashEq");
        BOOST_CHECK_NO_THROW(epec.findNashEq(true));
        BOOST_TEST_MESSAGE("Testing results:");
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 0), 100, 0.001);
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 0), 20, 0.01);
    }

    BOOST_AUTO_TEST_CASE(Bilevel_PriceCap1_test) {
        /** Testing a Single country (C1) with a single follower (F1)
     *  LeaderConstraints: price cap 299
     *  The price cap will enforce production to q=20 for the follower
     **/
        BOOST_TEST_MESSAGE("Testing a single bilevel problem with feasible price cap.");

        Models::FollPar FP;
        FP.capacities = {100};
        FP.costs_lin = {10};
        FP.costs_quad = {0.5};
        FP.emission_costs = {6};
        FP.names = {"Blu"};
        GRBEnv env = GRBEnv();
        arma::sp_mat TrCo(1, 1);
        TrCo(0, 0) = 0;

        Models::EPEC epec(&env);
        Models::LeadAllPar Country(FP.capacities.size(), "One", FP, {300, 0.05}, {-1, -1, -1, 299});
        BOOST_TEST_MESSAGE("MaxTax:20, PriceLimit:290 with alpha=300 and beta=0.05");
        BOOST_TEST_MESSAGE("PriceLimit coincides with domestic demand price");
        BOOST_TEST_MESSAGE("Expected: q=20;p=299");
        BOOST_TEST_MESSAGE("testing Models::addCountry");
        BOOST_CHECK_NO_THROW(epec.addCountry(Country));
        BOOST_TEST_MESSAGE("testing Models::addTranspCost");
        BOOST_CHECK_NO_THROW(epec.addTranspCosts(TrCo));
        BOOST_TEST_MESSAGE("testing Models::finalize");
        BOOST_CHECK_NO_THROW(epec.finalize());
        BOOST_TEST_MESSAGE("testing Models::make_country_QP");
        BOOST_CHECK_NO_THROW(epec.make_country_QP());
        BOOST_CHECK_NO_THROW(epec.testLCP(0));
        BOOST_TEST_MESSAGE("testing Models::findNashEq");
        BOOST_CHECK_NO_THROW(epec.findNashEq(true));
        BOOST_TEST_MESSAGE("Testing results:");
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 0), 20, 0.001);
    }

    BOOST_AUTO_TEST_CASE(Bilevel_PriceCap2_test) {
        /** Testing a Single country (C1) with a single follower (F1)
     *  LeaderConstraints: price cap 295
     *  The price cap is infeasible, hence we should get an exception in make_country_QP
     **/
        BOOST_TEST_MESSAGE("Testing a single bilevel problem with infeasible price cap.");
        Models::FollPar FP;
        FP.capacities = {100};
        FP.costs_lin = {10};
        FP.costs_quad = {0.5};
        FP.emission_costs = {6};
        FP.names = {"Blu"};
        GRBEnv env = GRBEnv();
        arma::sp_mat TrCo(1, 1);
        TrCo(0, 0) = 0;
        Models::LeadAllPar Country(FP.capacities.size(), "One", FP, {300, 0.05}, {20, -1, -1, 85});

        BOOST_TEST_MESSAGE("MaxTax:20, PriceLimit:85 with alpha=300 and beta=0.05");
        BOOST_TEST_MESSAGE("Expected: exception in make_country_QP (infeasability PriceLimit<295)");
        Models::EPEC epec(&env);
        BOOST_TEST_MESSAGE("testing Models::addCountry");
        BOOST_CHECK_NO_THROW(epec.addCountry(Country));
        BOOST_TEST_MESSAGE("testing Models::addTranspCost");
        BOOST_CHECK_NO_THROW(epec.addTranspCosts(TrCo));
        BOOST_TEST_MESSAGE("testing Models::finalize");
        BOOST_CHECK_NO_THROW(epec.finalize());
        BOOST_TEST_MESSAGE("testing Models::make_country_QP, expecting an exception :(");
        BOOST_CHECK_THROW(epec.make_country_QP(), string);
    }

    BOOST_AUTO_TEST_CASE(Bilevel_PriceCapTaxCap_test) {
        /** Testing a Single country (C1) with a single follower (F1)
     *  LeaderConstraints: price cap 295 and TaxCap at 20
     *  The price cap is feasible, hence we should expect max taxation (20) and q=20
     **/
        BOOST_TEST_MESSAGE("Testing a single bilevel problem with tax cap and price cap.");
        Models::FollPar FP;
        FP.capacities = {100};
        FP.costs_lin = {10};
        FP.costs_quad = {0.5};
        FP.emission_costs = {6};
        FP.names = {"Blu"};
        GRBEnv env = GRBEnv();
        arma::sp_mat TrCo(1, 1);
        TrCo(0, 0) = 0;
        Models::LeadAllPar Country(FP.capacities.size(), "One", FP, {300, 0.05}, {20, -1, -1, 295});

        Models::EPEC epec(&env);
        BOOST_TEST_MESSAGE("MaxTax:20, PriceLimit:290 with alpha=300 and beta=0.05");
        BOOST_TEST_MESSAGE("PriceLimit coincides with domestic demand price");
        BOOST_TEST_MESSAGE("Expected: q=100;t=20");
        BOOST_TEST_MESSAGE("testing Models::addCountry");
        BOOST_CHECK_NO_THROW(epec.addCountry(Country));
        BOOST_TEST_MESSAGE("testing Models::addTranspCost");
        BOOST_CHECK_NO_THROW(epec.addTranspCosts(TrCo));
        BOOST_TEST_MESSAGE("testing Models::finalize");
        BOOST_CHECK_NO_THROW(epec.finalize());
        BOOST_TEST_MESSAGE("testing Models::make_country_QP");
        BOOST_CHECK_NO_THROW(epec.make_country_QP());
        BOOST_TEST_MESSAGE("testing Models::findNashEq");
        BOOST_CHECK_NO_THROW(epec.findNashEq(true));
        BOOST_TEST_MESSAGE("Testing results:");
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 0), 100, 0.001);
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 0), 20, 0.01);

    }

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Models_C1Fn__Tests)

    /* This test suite perform unit tests for EPEC problems with one country and multiple followers
     */

    BOOST_AUTO_TEST_CASE(C1F1_test) {
        /** Testing a Single country (C1) with a single follower (F1)
        *  LeaderConstraints: price cap 300 and tax cap 100
        *  The follower with the lowest marginal cost will produce more
         * taxation will be maximized to the cap for both followers
        **/
        BOOST_TEST_MESSAGE("Testing 2Followers 1 Country with tax cap and price cap.");
        Models::FollPar FP;
        FP.capacities = {100, 200};
        FP.costs_lin = {10, 4};
        FP.costs_quad = {5, 3};
        FP.emission_costs = {6, 10};
        FP.names = {"Rosso", "Bianco"};
        Models::LeadAllPar Country(FP.capacities.size(), "One", FP, {300, 0.05}, {100, -1, -1, 300});
        GRBEnv env = GRBEnv();
        arma::sp_mat TrCo(1, 1);
        TrCo(0, 0) = 0;

        BOOST_TEST_MESSAGE("MaxTax:100, PriceLimit:300 with alpha=300 and beta=0.05");
        BOOST_TEST_MESSAGE("Expected: margCost(Rosso)>margCost(Bianco);t_0=t_1=maxTax=100");
        Models::EPEC epec(&env);
        BOOST_TEST_MESSAGE("testing Models::addCountry");
        BOOST_CHECK_NO_THROW(epec.addCountry(Country));
        BOOST_TEST_MESSAGE("testing Models::addTranspCost");
        BOOST_CHECK_NO_THROW(epec.addTranspCosts(TrCo));
        BOOST_TEST_MESSAGE("testing Models::finalize");
        BOOST_CHECK_NO_THROW(epec.finalize());
        BOOST_TEST_MESSAGE("testing Models::make_country_QP");
        BOOST_CHECK_NO_THROW(epec.make_country_QP());
        BOOST_TEST_MESSAGE("testing Models::findNashEq");
        BOOST_CHECK_NO_THROW(epec.findNashEq(true));
        double margRosso = FP.costs_quad[0] * epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 0) +
                           FP.costs_lin[0] + epec.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 0);
        double margBianco = FP.costs_quad[1] * epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 1) +
                            FP.costs_lin[1] + epec.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 1);
        BOOST_CHECK_MESSAGE(margRosso > margBianco, "Checking marginal cost of Rosso > marginal cost of Bianco");
        BOOST_CHECK_MESSAGE(epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 1) >
                            epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 0),
                            "Checking q_Rosso<q_Bianco");
        BOOST_TEST_MESSAGE("checking taxation on Rosso & Bianco");
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 0), 100, 0.01);
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 1), 100, 0.01);

    }

    BOOST_AUTO_TEST_CASE(C1F1_Capacities_test) {
        /** Testing a Single country (C1) with a single follower (F1)
        *  LeaderConstraints: price cap 300 and tax cap 100
        *  The follower with the lowest marginal cost will produce more
         * taxation will be maximized to the cap for both followers
         * Also, the second follower will produce up to its capacity (50)
        **/
        BOOST_TEST_MESSAGE("Testing 2Followers (one with cap) 1 Country with tax cap and price cap.");
        Models::FollPar FP;
        FP.capacities = {100, 50};
        FP.costs_lin = {10, 4};
        FP.costs_quad = {5, 3};
        FP.emission_costs = {6, 10};
        FP.names = {"Rosso", "Bianco"};
        Models::LeadAllPar Country(FP.capacities.size(), "One", FP, {300, 0.05}, {100, -1, -1, 300});
        GRBEnv env = GRBEnv();
        arma::sp_mat TrCo(1, 1);
        TrCo(0, 0) = 0;
        BOOST_TEST_MESSAGE("MaxTax:100, PriceLimit:300 with alpha=300 and beta=0.05");
        BOOST_TEST_MESSAGE("Expected: margCost(Rosso)>margCost(Bianco);t_0=t_1=maxTax=100");
        BOOST_TEST_MESSAGE("Expected: maximum production of Bianco (q1=50), Rosso producing slightly more than before");

        Models::EPEC epec(&env);
        BOOST_TEST_MESSAGE("testing Models::addCountry");
        BOOST_CHECK_NO_THROW(epec.addCountry(Country));
        BOOST_TEST_MESSAGE("testing Models::addTranspCost");
        BOOST_CHECK_NO_THROW(epec.addTranspCosts(TrCo));
        BOOST_TEST_MESSAGE("testing Models::finalize");
        BOOST_CHECK_NO_THROW(epec.finalize());
        BOOST_TEST_MESSAGE("testing Models::make_country_QP");
        BOOST_CHECK_NO_THROW(epec.make_country_QP());
        BOOST_TEST_MESSAGE("testing Models::findNashEq");
        BOOST_CHECK_NO_THROW(epec.findNashEq(true));
        BOOST_TEST_MESSAGE("checking production on Bianco");
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 1), 50, 0.01);
        BOOST_CHECK_MESSAGE(epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 1) >
                            epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 0),
                            "checking production on Rosso");
        BOOST_TEST_MESSAGE("checking taxation on Rosso & Bianco");
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 0), 100, 0.01);
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 1), 100, 0.01);
    }

    BOOST_AUTO_TEST_CASE(C1F5_test) {
        /** Testing a Single country (C1) with a 5 followers (F5)
        *  LeaderConstraints:  tax cap 100
        *  The followers with the lowest marginal cost will produce more
         * taxation will be maximized to the cap for all followers
         * Also, the more economical followers will produce up to their cap
        **/
        BOOST_TEST_MESSAGE("Testing 5Followers 1 Country.");
        BOOST_TEST_MESSAGE("MaxTax:100,  with alpha=400 and beta=0.05");
        BOOST_TEST_MESSAGE("Expected: MaxTax on all followers and maximum production for polluting ones. ");
        Models::FollPar FP5;
        GRBEnv env = GRBEnv();
        arma::sp_mat TrCo(1, 1);
        TrCo(0, 0) = 0;
        FP5.capacities = {100, 70, 50, 30, 20};
        FP5.costs_lin = {20, 15, 13, 10, 5};
        FP5.costs_quad = {5, 4, 3, 3, 2};
        FP5.emission_costs = {2, 4, 9, 10, 15};
        FP5.names = {"Rosso", "Bianco", "Blu", "Viola", "Verde"};
        Models::LeadAllPar Country5(FP5.capacities.size(), "One", FP5, {400, 0.05}, {100, -1, -1, -1});
        Models::EPEC epec(&env);
        BOOST_TEST_MESSAGE("testing Models::addCountry");
        BOOST_CHECK_NO_THROW(epec.addCountry(Country5));
        BOOST_TEST_MESSAGE("testing Models::addTranspCost");
        BOOST_CHECK_NO_THROW(epec.addTranspCosts(TrCo));
        BOOST_TEST_MESSAGE("testing Models::finalize");
        BOOST_CHECK_NO_THROW(epec.finalize());
        BOOST_TEST_MESSAGE("testing Models::make_country_QP");
        BOOST_CHECK_NO_THROW(epec.make_country_QP());
        BOOST_TEST_MESSAGE("testing Models::findNashEq");
        BOOST_CHECK_NO_THROW(epec.findNashEq(true));
        BOOST_TEST_MESSAGE("checking taxation");
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 0), 100, 0.01);
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 1), 100, 0.01);
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 2), 100, 0.01);
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 3), 100, 0.01);
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 4), 100, 0.01);
        BOOST_TEST_MESSAGE("checking production of polluting followers");
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 2), 50, 0.01);
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 3), 30, 0.01);
        BOOST_CHECK_CLOSE(epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 4), 20, 0.01);
    }

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(Models_CnFn__Tests)

    /* This test suite perform basic unit tests for generalized EPEC problem with multiple countries and followers
     */

    BOOST_AUTO_TEST_CASE(C2F1_test) {
        /** Testing two countries (C2) with a single follower (F1)
        *  LeaderConstraints: price cap 300 and tax cap 100
        *  The follower with the lowest marginal cost will produce more
        **/
        BOOST_TEST_MESSAGE("Testing 2 Countries with a follower each -  with tax cap and price cap.");
        Models::FollPar FP;
        FP.capacities = {550};
        FP.costs_lin = {4};
        FP.costs_quad = {3};
        FP.emission_costs = {10};
        FP.names = {"Rosso"};
        Models::LeadAllPar Country0(FP.capacities.size(), "One", FP, {300, 0.7}, {100, -1, -1, 299.99});
        Models::LeadAllPar Country1(FP.capacities.size(), "Two", FP, {350, 0.5}, {100, -1, -1, -1});
        GRBEnv env = GRBEnv();
        arma::sp_mat TrCo(2, 2);
        TrCo.zeros(2,2);
        TrCo(0, 1) = 1;
        TrCo(1, 0) = TrCo(0, 1);

        BOOST_TEST_MESSAGE("MaxTax:100, PriceLimit:300 with alpha=300 and beta=0.05");
        //BOOST_TEST_MESSAGE("Expected: margCost(Rosso)>margCost(Bianco);t_0=t_1=maxTax=100");
        Models::EPEC epec(&env);
        BOOST_TEST_MESSAGE("testing Models::addCountry (0)");
        BOOST_CHECK_NO_THROW(epec.addCountry(Country0));
        BOOST_TEST_MESSAGE("testing Models::addCountry (1)");
        BOOST_CHECK_NO_THROW(epec.addCountry(Country1));
        BOOST_TEST_MESSAGE("testing Models::addTranspCost");
        BOOST_CHECK_NO_THROW(epec.addTranspCosts(TrCo));
        BOOST_TEST_MESSAGE("testing Models::finalize");
        BOOST_CHECK_NO_THROW(epec.finalize());
        BOOST_TEST_MESSAGE("testing Models::make_country_QP");
        BOOST_CHECK_NO_THROW(epec.make_country_QP());
        BOOST_TEST_MESSAGE("testing Models::findNashEq");
        BOOST_CHECK_NO_THROW(epec.findNashEq(true));
        double margCountryOne = FP.costs_quad[0] * epec.x.at(epec.getPosition(0, Models::LeaderVars::FollowerStart) + 0) +
                           FP.costs_lin[0] + epec.x.at(epec.getPosition(0, Models::LeaderVars::Tax) + 0);
        double margCountryTwo = FP.costs_quad[0] * epec.x.at(epec.getPosition(1, Models::LeaderVars::FollowerStart) + 0) +
                            FP.costs_lin[0] + epec.x.at(epec.getPosition(1, Models::LeaderVars::Tax) + 0);
        cout << margCountryOne << "\t" << margCountryTwo << endl;

    }


BOOST_AUTO_TEST_SUITE_END()

