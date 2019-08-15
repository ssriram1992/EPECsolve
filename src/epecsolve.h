#ifndef EPECSOLVE_H
#define EPECSOLVE_H

#define VERBOSE false

#include <map>
#include <iostream>
#include <vector>
#include<armadillo>

using namespace std;

using perps = vector<pair<unsigned int, unsigned int>>;

ostream &operator<<(ostream &ost, perps C);

inline bool operator<(vector<int> Fix1, vector<int> Fix2);

inline bool operator==(vector<int> Fix1, vector<int> Fix2);

template<class T>
ostream &operator<<(ostream &ost, vector<T> v);

template<class T, class S>
ostream &operator<<(ostream &ost, pair<T, S> p);

// Forward declarations
namespace Game {
    struct QP_objective;
    struct QP_constraints;

    class MP_Param;

    class QP_Param;

    class NashGame;

    class LCP;
}; // namespace Game
namespace Models {
    class EPEC;
};

#include "games.h"
#include "lcptolp.h"
#include "models.h"
#include "utils.h"

#endif
