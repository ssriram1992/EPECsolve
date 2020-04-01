#pragma once

/** @file src/epecsolve.h Forward declarations
 */

#define VERBOSE false
#define EPECVERSION_MAJOR "2.0"
#define EPECVERSION_MINOR "alpha"

#include <armadillo>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

using perps = std::vector<std::pair<unsigned int, unsigned int>>;
std::ostream &operator<<(std::ostream &ost, perps C);
template <class T>
std::ostream &operator<<(std::ostream &ost, std::vector<T> v);
template <class T, class S>
std::ostream &operator<<(std::ostream &ost, std::pair<T, S> p);
using spmat_Vec = std::vector<std::unique_ptr<arma::sp_mat>>;
using vec_Vec = std::vector<std::unique_ptr<arma::vec>>;

// Forward declarations
namespace Game {
struct QP_Objective;
struct QP_Constraints;
class MP_Param;
class QP_Param;
class NashGame;
class LCP;
class PolyLCP;
class OuterLCP;
class EPEC;
enum class EPECAddPolyMethod {
  Sequential,        ///< Adds polyhedra by selecting them in order
  ReverseSequential, ///< Adds polyhedra by selecting them in reverse
                     ///< Sequential order
  Random ///< Adds the next polyhedron by selecting Random feasible one
};

} // namespace Game
namespace Algorithms {
// Forward declarations
class FullEnumeration;
class InnerApproximation;
class CombinatorialPNE;
class OuterApproximation;
class PolyBase;
} // namespace Algorithms

#include "LCP/lcp.h"
#include "games.h"
#include "utils.h"
