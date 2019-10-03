#pragma once

#define VERBOSE false
#define EPECVERSION 0.1

#include <armadillo>
#include <iostream>
#include <map>
#include <vector>

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
class EPEC;
enum class EPECAddPolyMethod {
  sequential,         ///< Adds polyhedra by selecting them in order
  reverse_sequential, ///< Adds polyhedra by selecting them in reverse
                      ///< sequential order
  random ///< Adds the next polyhedra by selecting random feasible one
};

} // namespace Game
namespace Models {
class EPEC;
}

#include "games.h"
#include "lcptolp.h"
#include "utils.h"
