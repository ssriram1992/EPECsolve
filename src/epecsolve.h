#pragma once

/** @file src/epecsolve.h Forward declarations
 */

#define VERBOSE false
#define EPECVERSION 2.0

#include <armadillo>
#include <memory>
#include <vector>
#include <set>
#include <string>
#include <boost/log/trivial.hpp>
#include <gurobi_c++.h>

using perps = std::vector<std::pair<unsigned int, unsigned int>>;
std::ostream &operator<<(std::ostream &ost, perps C);
inline bool operator<(std::vector<short int> Fix1, std::vector<short int> Fix2);
inline bool operator==(std::vector<short int> Fix1,
                       std::vector<short int> Fix2);
template <class T>
std::ostream &operator<<(std::ostream &ost, std::vector<T> v);
template <class T, class S>
std::ostream &operator<<(std::ostream &ost, std::pair<T, S> p);

// Forward declarations
namespace Game {
struct QP_objective;
struct QP_constraints;
class MP_Param;
class QP_Param;
class NashGame;
class LCP;
class PolyLCP;
class EPEC;
enum class EPECAddPolyMethod {
  sequential,         ///< Adds polyhedra by selecting them in order
  reverse_sequential, ///< Adds polyhedra by selecting them in reverse
                      ///< sequential order
  random ///< Adds the next polyhedra by selecting random feasible one
};

} // namespace Game
namespace Algorithms {
// Forward declarations
class fullEnumeration;
class innerApproximation;
class combinatorialPNE;
class outerApproximation;
class PolyBase;
} // namespace Algorithms

#include "LCP/LCP.h"
#include "games.h"
#include "utils.h"
