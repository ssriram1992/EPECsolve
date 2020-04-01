#pragma once

#include "../src/algorithms/algorithms.h"
#include "../src/games.h"
#include "../src/models.h"
#include "LCP/lcp.h"
#include <armadillo>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <gurobi_c++.h>
#include <iomanip>
#include <iostream>
#include <random>

#define BOOST_TEST_MODULE EPECTest

#include <boost/test/unit_test.hpp>

#include "core_tests.h"
#include "testdata.h"
