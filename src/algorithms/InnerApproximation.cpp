#include "algorithms/innerapproximation.h"
#include "algorithms/combinatorialpne.h"
#include "epecsolve.h"

#include <armadillo>
#include <boost/log/trivial.hpp>
#include <chrono>
#include <gurobi_c++.h>
#include <set>
#include <string>

void Algorithms::InnerApproximation::solve() {
  /**
   * Wraps the Algorithm with the postSolving operations
   */
  this->start();
  this->postSolving();
}

void Algorithms::InnerApproximation::start() {
  /**
   * Given the referenced EPEC instance, this method solves it through the inner
   * approximation Algorithm.
   */
  // Set the initial point for all countries as 0 and solve the respective LCPs?
  this->EPECObject->SolutionX.zeros(this->EPECObject->NumVariables);
  bool solved = {false};
  bool addRandPoly{false};
  bool infeasCheck{false};
  // When true, a MNE has been found. The algorithm now tries to find a PNE, at
  // the cost of incrementally enumerating the remaining polyhedra, up to the
  // TimeLimit (if any).
  //
  bool incrementalEnumeration{false};

  std::vector<arma::vec> prevDevns(this->EPECObject->NumPlayers);
  this->EPECObject->Stats.NumIterations = 0;
  if (this->EPECObject->Stats.AlgorithmParam.AddPolyMethod ==
      Game::EPECAddPolyMethod::Random) {
    for (unsigned int i = 0; i < this->EPECObject->NumPlayers; ++i) {
      // 42 is the answer, we all know
      long int seed =
          this->EPECObject->Stats.AlgorithmParam.AddPolyMethodSeed < 0
              ? std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count() +
                    42 + PolyLCP.at(i)->getNumRows()
              : this->EPECObject->Stats.AlgorithmParam.AddPolyMethodSeed;
      PolyLCP.at(i)->AddPolyMethodSeed = seed;
    }
  }
  if (this->EPECObject->Stats.AlgorithmParam.TimeLimit > 0)
    this->EPECObject->InitTime = std::chrono::high_resolution_clock::now();

  // Stay in this loop, till you find a Nash equilibrium or prove that there
  // does not exist a Nash equilibrium or you run out of time.
  while (!solved) {
    ++this->EPECObject->Stats.NumIterations;
    BOOST_LOG_TRIVIAL(info)
        << "Algorithms::InnerApproximation::solve: Iteration "
        << std::to_string(this->EPECObject->Stats.NumIterations);

    if (addRandPoly) {
      BOOST_LOG_TRIVIAL(info) << "Algorithms::InnerApproximation::solve: using "
                                 "heuristical polyhedra selection";
      bool success = this->addRandomPoly2All(
          this->EPECObject->Stats.AlgorithmParam.Aggressiveness,
          this->EPECObject->Stats.NumIterations == 1);
      if (!success) {
        this->EPECObject->Stats.Status = Game::EPECsolveStatus::NashEqNotFound;
        return;
      }
    } else { // else we are in the case of finding deviations.
      unsigned int deviatedCountry{0};
      arma::vec countryDeviation{};
      if (this->isSolved(&deviatedCountry, &countryDeviation)) {
        this->EPECObject->Stats.Status = Game::EPECsolveStatus::NashEqFound;
        this->EPECObject->Stats.PureNashEquilibrium =
            this->isPureStrategy();
        if ((this->EPECObject->Stats.AlgorithmParam.PureNashEquilibrium &&
             !this->EPECObject->Stats.PureNashEquilibrium)) {
          // We are seeking for a pure strategy. Then, here we switch between an
          // incremental
          // enumeration or combinations of pure strategies.
          if (this->EPECObject->Stats.AlgorithmParam.RecoverStrategy ==
              Game::EPECRecoverStrategy::IncrementalEnumeration) {
            BOOST_LOG_TRIVIAL(info) << "Algorithms::InnerApproximation::solve: "
                                       "triggering recover strategy "
                                       "(IncrementalEnumeration)";
            incrementalEnumeration = true;
          } else if (this->EPECObject->Stats.AlgorithmParam.RecoverStrategy ==
                     Game::EPECRecoverStrategy::Combinatorial) {
            BOOST_LOG_TRIVIAL(info)
                << "Algorithms::InnerApproximation::solve: triggering "
                   "recover strategy (Combinatorial)";
            // In this case, we want to try all the combinations of pure
            // strategies, except the ones between polyhedra we already tested.
            std::vector<std::set<unsigned long int>> excludeList;
            for (unsigned long j = 0; j < this->EPECObject->NumPlayers; ++j) {
              excludeList.push_back(PolyLCP.at(j)->getAllPolyhedra());
            }
            Algorithms::CombinatorialPNE combPNE(this->Env, this->EPECObject);
            combPNE.solveWithExcluded(excludeList);
            return;
          }

        } else
          return;
      }
      // Vector of deviations for the countries
      std::vector<arma::vec> devns =
          std::vector<arma::vec>(this->EPECObject->NumPlayers);
      this->getAllDeviations(devns, this->EPECObject->SolutionX, prevDevns);
      prevDevns = devns;
      unsigned int addedPoly = this->addDeviatedPolyhedron(devns, infeasCheck);
      if (addedPoly == 0 && this->EPECObject->Stats.NumIterations > 1 &&
          !incrementalEnumeration) {
        BOOST_LOG_TRIVIAL(error)
            << " In Algorithms::InnerApproximation::solve: Not "
               "Solved, but no deviation? Error!\n This might be due to "
               "Numerical issues (tolerances)";
        this->EPECObject->Stats.Status = Game::EPECsolveStatus::Numerical;
        solved = true;
      }
      if (infeasCheck && this->EPECObject->Stats.NumIterations == 1) {
        BOOST_LOG_TRIVIAL(warning)
            << " In Algorithms::InnerApproximation::solve: Problem is "
               "infeasible";
        this->EPECObject->Stats.Status = Game::EPECsolveStatus::NashEqNotFound;
        return;
      }
    }

    this->EPECObject->makePlayersQPs();

    // TimeLimit
    if (this->EPECObject->Stats.AlgorithmParam.TimeLimit > 0) {
      const std::chrono::duration<double> timeElapsed =
          std::chrono::high_resolution_clock::now() -
          this->EPECObject->InitTime;
      const double timeRemaining =
          this->EPECObject->Stats.AlgorithmParam.TimeLimit -
          timeElapsed.count();
      addRandPoly =
          !this->EPECObject->computeNashEq(
              this->EPECObject->Stats.AlgorithmParam.PureNashEquilibrium,
              timeRemaining) &&
          !incrementalEnumeration;
    } else {
      // No Time Limit
      addRandPoly =
          !this->EPECObject->computeNashEq(
              this->EPECObject->Stats.AlgorithmParam.PureNashEquilibrium) &&
          !incrementalEnumeration;
    }
    if (addRandPoly)
      this->EPECObject->Stats.LostIntermediateEq++;
    for (unsigned int i = 0; i < this->EPECObject->NumPlayers; ++i) {
      BOOST_LOG_TRIVIAL(info)
          << "Country " << i << PolyLCP.at(i)->feasabilityDetailString();
    }
    // This might be reached when a NashEq is found, and need to be verified.
    // Anyway, we are over the TimeLimit and we should stop
    if (this->EPECObject->Stats.AlgorithmParam.TimeLimit > 0) {
      const std::chrono::duration<double> timeElapsed =
          std::chrono::high_resolution_clock::now() -
          this->EPECObject->InitTime;
      const double timeRemaining =
          this->EPECObject->Stats.AlgorithmParam.TimeLimit -
          timeElapsed.count();
      if (timeRemaining <= 0) {
        if (!incrementalEnumeration)
          this->EPECObject->Stats.Status = Game::EPECsolveStatus::TimeLimit;
        return;
      }
    }
  }
}

bool Algorithms::InnerApproximation::addRandomPoly2All(
    unsigned int aggressiveLevel, bool stopOnSingleInfeasibility)
/**
 * Makes a call to to Game::LCP::addAPoly for each member in
 * Game::EPEC::PlayersLCP and tries to add a polyhedron to get a better inner
 * approximation for the LCP. @p aggressiveLevel is the maximum number of
 * polyhedra it will try to add to each country. Setting it to an arbitrarily
 * high value will mimic complete enumeration.
 *
 * If @p stopOnSingleInfeasibility is true, then the function returns false and
 * aborts all operation as soon as it finds that it cannot add polyhedra to some
 * country. On the other hand if @p stopOnSingleInfeasibility is false, the
 * function returns false, only if it is not possible to add polyhedra to
 * <i>any</i> of the countries.
 * @returns true if successfully added the maximum possible number of polyhedra
 * not greater than aggressiveLevel.
 */
{
  BOOST_LOG_TRIVIAL(trace) << "Adding Random polyhedra to countries";
  bool infeasible{true};
  for (unsigned int i = 0; i < this->EPECObject->NumPlayers; i++) {
    auto addedPolySet = PolyLCP.at(i)->addAPoly(
        aggressiveLevel, this->EPECObject->Stats.AlgorithmParam.AddPolyMethod);
    if (stopOnSingleInfeasibility && addedPolySet.empty()) {
      BOOST_LOG_TRIVIAL(info)
          << "Algorithms::InnerApproximation::addRandomPoly2All: No Nash "
             "equilibrium. due to "
             "infeasibility of country "
          << i;
      return false;
    }
    if (!addedPolySet.empty())
      infeasible = false;
  }
  return !infeasible;
}

bool Algorithms::InnerApproximation::getAllDeviations(
    std::vector<arma::vec>
        &deviations, ///< [out] The vector of deviations for all players
    const arma::vec &guessSol, ///< [in] The guess for the solution vector
    const std::vector<arma::vec>
        &prevDev //<[in] The previous vector of deviations, if any exist.
) const
/**
 * @brief Given a potential solution vector, returns a profitable deviation (if
 * it exists) for all players. @param
 * @return a vector of computed deviations, which empty if at least one
 * deviation cannot be computed
 * @param prevDev can be empty
 */
{
  deviations = std::vector<arma::vec>(this->EPECObject->NumPlayers);

  for (unsigned int i = 0; i < this->EPECObject->NumPlayers;
       ++i) { // For each country
    // If we cannot compute a deviation, it means model is infeasible!
    if (this->EPECObject->respondSol(deviations.at(i), i, guessSol,
                                     prevDev.at(i)) == GRB_INFINITY)
      return false;
    // cout << "Game::EPEC::getAllDeviations: deviations(i): "
    // <<deviations.at(i);
  }
  return true;
}

unsigned int Algorithms::InnerApproximation::addDeviatedPolyhedron(
    const std::vector<arma::vec>
        &deviations, ///< devns.at(i) is a profitable deviation
    ///< for the i-th country from the current this->SolutionX
    bool &infeasCheck ///< Useful for the first iteration of iterativeNash. If
                      ///< true, at least one player has no polyhedron that can
                      ///< be added. In the first iteration, this translates to
                      ///< x
) const {
  /**
   * Given a profitable deviation for each country, adds <i>a</i> polyhedron in
   * the feasible region of each country to the corresponding country's
   * Game::LCP object (this->PlayersLCP.at(i)) 's vector of feasible
   * polyhedra.
   *
   * Naturally, this makes the inner approximation of the Game::LCP better, by
   * including one additional polyhedron.
   */

  infeasCheck = false;
  unsigned int added = 0;
  for (unsigned int i = 0; i < this->EPECObject->NumPlayers;
       ++i) { // For each country
    bool ret = false;
    if (!deviations.at(i).empty())
      PolyLCP.at(i)->addPolyFromX(deviations.at(i), ret);
    if (ret) {
      BOOST_LOG_TRIVIAL(trace)
          << "Algorithms::InnerApproximation::addDeviatedPolyhedron: added "
             "polyhedron for player "
          << i;
      ++added;
    } else {
      infeasCheck = true;
      BOOST_LOG_TRIVIAL(trace)
          << "Algorithms::InnerApproximation::addDeviatedPolyhedron: NO "
             "polyhedron added for player "
          << i;
    }
  }
  return added;
}