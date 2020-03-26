#include "innerApproximation.h"

using namespace std;
void Algorithms::innerApproximation::solve() {
  /**
   * Wraps the algorithm with the postSolving operations
   */
  this->start();
  this->postSolving();
}

void Algorithms::innerApproximation::start() {
  /**
   * Given the referenced EPEC instance, this method solves it through the inner
   * approximation algorithm.
   */
  // Set the initial point for all countries as 0 and solve the respective LCPs?
  this->EPECObject->sol_x.zeros(this->EPECObject->nVarinEPEC);
  bool solved = {false};
  bool addRandPoly{false};
  bool infeasCheck{false};
  // When true, a MNE has been found. The algorithm now tries to find a PNE, at
  // the cost of incrementally enumerating the remaining polyhedra, up to the
  // TimeLimit (if any).
  //
  bool incrementalEnumeration{false};

  std::vector<arma::vec> prevDevns(this->EPECObject->nCountr);
  this->EPECObject->Stats.numIteration = 0;
  if (this->EPECObject->Stats.AlgorithmParam.addPolyMethod ==
      EPECAddPolyMethod::random) {
    for (unsigned int i = 0; i < this->EPECObject->nCountr; ++i) {
      // 42 is the answer, we all know
      long int seed =
          this->EPECObject->Stats.AlgorithmParam.addPolyMethodSeed < 0
              ? chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count() +
                    42 + poly_LCP.at(i)->getNrow()
              : this->EPECObject->Stats.AlgorithmParam.addPolyMethodSeed;
      poly_LCP.at(i)->addPolyMethodSeed = seed;
    }
  }
  if (this->EPECObject->Stats.AlgorithmParam.timeLimit > 0)
    this->EPECObject->initTime = std::chrono::high_resolution_clock::now();

  // Stay in this loop, till you find a Nash equilibrium or prove that there
  // does not exist a Nash equilibrium or you run out of time.
  while (!solved) {
    ++this->EPECObject->Stats.numIteration;
    BOOST_LOG_TRIVIAL(info)
        << "Algorithms::innerApproximation::solve: Iteration "
        << to_string(this->EPECObject->Stats.numIteration);

    if (addRandPoly) {
      BOOST_LOG_TRIVIAL(info) << "Algorithms::innerApproximation::solve: using "
                                 "heuristical polyhedra selection";
      bool success = this->addRandomPoly2All(
          this->EPECObject->Stats.AlgorithmParam.aggressiveness,
          this->EPECObject->Stats.numIteration == 1);
      if (!success) {
        this->EPECObject->Stats.status = Game::EPECsolveStatus::nashEqNotFound;
        solved = true;
        return;
      }
    } else { // else we are in the case of finding deviations.
      unsigned int deviatedCountry{0};
      arma::vec countryDeviation{};
      if (this->EPECObject->isSolved(&deviatedCountry, &countryDeviation)) {
        this->EPECObject->Stats.status = Game::EPECsolveStatus::nashEqFound;
        this->EPECObject->Stats.pureNE = this->EPECObject->isPureStrategy();
        if ((this->EPECObject->Stats.AlgorithmParam.pureNE &&
             !this->EPECObject->Stats.pureNE)) {
          // We are seeking for a pure strategy. Then, here we switch between an
          // incremental
          // enumeration or combinations of pure strategies.
          if (this->EPECObject->Stats.AlgorithmParam.recoverStrategy ==
              Game::EPECRecoverStrategy::incrementalEnumeration) {
            BOOST_LOG_TRIVIAL(info) << "Algorithms::innerApproximation::solve: "
                                       "triggering recover strategy "
                                       "(incrementalEnumeration)";
            incrementalEnumeration = true;
          } else if (this->EPECObject->Stats.AlgorithmParam.recoverStrategy ==
                     Game::EPECRecoverStrategy::combinatorial) {
            BOOST_LOG_TRIVIAL(info)
                << "Algorithms::innerApproximation::solve: triggering "
                   "recover strategy (combinatorial)";
            // In this case, we want to try all the combinations of pure
            // strategies, except the ones between polyhedra we already tested.
            std::vector<std::set<unsigned long int>> excludeList;
            for (int j = 0; j < this->EPECObject->nCountr; ++j) {
              excludeList.push_back(poly_LCP.at(j)->getAllPolyhedra());
            }
            Algorithms::combinatorialPNE combPNE(this->env, this->EPECObject);
            combPNE.solve(excludeList);
            return;
          }

        } else {
          solved = true;
          return;
        }
      }
      // Vector of deviations for the countries
      std::vector<arma::vec> devns =
          std::vector<arma::vec>(this->EPECObject->nCountr);
      this->getAllDevns(devns, this->EPECObject->sol_x, prevDevns);
      prevDevns = devns;
      unsigned int addedPoly = this->addDeviatedPolyhedron(devns, infeasCheck);
      if (addedPoly == 0 && this->EPECObject->Stats.numIteration > 1 &&
          !incrementalEnumeration) {
        BOOST_LOG_TRIVIAL(error)
            << " In Algorithms::innerApproximation::solve: Not "
               "Solved, but no deviation? Error!\n This might be due to "
               "numerical issues (tollerances)";
        this->EPECObject->Stats.status = EPECsolveStatus::numerical;
        solved = true;
      }
      if (infeasCheck && this->EPECObject->Stats.numIteration == 1) {
        BOOST_LOG_TRIVIAL(warning)
            << " In Algorithms::innerApproximation::solve: Problem is "
               "infeasible";
        this->EPECObject->Stats.status = EPECsolveStatus::nashEqNotFound;
        solved = true;
        return;
      }
    }

    this->EPECObject->make_country_QP();

    // TimeLimit
    if (this->EPECObject->Stats.AlgorithmParam.timeLimit > 0) {
      const std::chrono::duration<double> timeElapsed =
          std::chrono::high_resolution_clock::now() -
          this->EPECObject->initTime;
      const double timeRemaining =
          this->EPECObject->Stats.AlgorithmParam.timeLimit -
          timeElapsed.count();
      addRandPoly =
          !this->EPECObject->computeNashEq(
              this->EPECObject->Stats.AlgorithmParam.pureNE, timeRemaining) &&
          !incrementalEnumeration;
    } else {
      // No Time Limit
      addRandPoly = !this->EPECObject->computeNashEq(
                        this->EPECObject->Stats.AlgorithmParam.pureNE) &&
                    !incrementalEnumeration;
    }
    if (addRandPoly)
      this->EPECObject->Stats.lostIntermediateEq++;
    for (unsigned int i = 0; i < this->EPECObject->nCountr; ++i) {
      BOOST_LOG_TRIVIAL(info)
          << "Country " << i << poly_LCP.at(i)->feas_detail_str();
    }
    // This might be reached when a NashEq is found, and need to be verified.
    // Anyway, we are over the timeLimit and we should stop
    if (this->EPECObject->Stats.AlgorithmParam.timeLimit > 0) {
      const std::chrono::duration<double> timeElapsed =
          std::chrono::high_resolution_clock::now() -
          this->EPECObject->initTime;
      const double timeRemaining =
          this->EPECObject->Stats.AlgorithmParam.timeLimit -
          timeElapsed.count();
      if (timeRemaining <= 0) {
        solved = false;
        if (!incrementalEnumeration)
          this->EPECObject->Stats.status = Game::EPECsolveStatus::timeLimit;
        return;
      }
    }
  }
}

bool Algorithms::innerApproximation::addRandomPoly2All(
    unsigned int aggressiveLevel, bool stopOnSingleInfeasibility)
/**
 * Makes a call to to Game::LCP::addAPoly for each member in
 * Game::EPEC::countries_LCP and tries to add a polyhedron to get a better inner
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
  BOOST_LOG_TRIVIAL(trace) << "Adding random polyhedra to countries";
  bool infeasible{true};
  for (unsigned int i = 0; i < this->EPECObject->nCountr; i++) {
    auto addedPolySet = poly_LCP.at(i)->addAPoly(
        aggressiveLevel, this->EPECObject->Stats.AlgorithmParam.addPolyMethod);
    if (stopOnSingleInfeasibility && addedPolySet.empty()) {
      BOOST_LOG_TRIVIAL(info)
          << "Algorithms::innerApproximation::addRandomPoly2All: No Nash "
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

bool Algorithms::innerApproximation::getAllDevns(
    std::vector<arma::vec>
        &devns, ///< [out] The vector of deviations for all players
    const arma::vec &guessSol, ///< [in] The guess for the solution vector
    const std::vector<arma::vec>
        &prevDev //<[in] The previous vecrtor of deviations, if any exist.
) const
/**
 * @brief Given a potential solution vector, returns a profitable deviation (if
 * it exists) for all players. @param
 * @return a vector of computed deviations, which empty if at least one
 * deviation cannot be computed
 * @param prevDev can be empty
 */
{
  devns = std::vector<arma::vec>(this->EPECObject->nCountr);

  for (unsigned int i = 0; i < this->EPECObject->nCountr;
       ++i) { // For each country
    // If we cannot compute a deviation, it means model is infeasible!
    if (this->EPECObject->RespondSol(devns.at(i), i, guessSol, prevDev.at(i)) ==
        GRB_INFINITY)
      return false;
    // cout << "Game::EPEC::getAllDevns: devns(i): " <<devns.at(i);
  }
  return true;
}

unsigned int Algorithms::innerApproximation::addDeviatedPolyhedron(
    const std::vector<arma::vec>
        &devns, ///< devns.at(i) is a profitable deviation
    ///< for the i-th country from the current this->sol_x
    bool &infeasCheck ///< Useful for the first iteration of iterativeNash. If
                      ///< true, at least one player has no polyhedron that can
                      ///< be added. In the first iteration, this translates to
                      ///< infeasability
) const {
  /**
   * Given a profitable deviation for each country, adds <i>a</i> polyhedron in
   * the feasible region of each country to the corresponding country's
   * Game::LCP object (this->countries_LCP.at(i)) 's vector of feasible
   * polyhedra.
   *
   * Naturally, this makes the inner approximation of the Game::LCP better, by
   * including one additional polyhedron.
   */

  infeasCheck = false;
  unsigned int added = 0;
  for (unsigned int i = 0; i < this->EPECObject->nCountr;
       ++i) { // For each country
    bool ret = false;
    if (!devns.at(i).empty())
      poly_LCP.at(i)->addPolyFromX(devns.at(i), ret);
    if (ret) {
      BOOST_LOG_TRIVIAL(trace)
          << "Algorithms::innerApproximation::addDeviatedPolyhedron: added "
             "polyhedron for player "
          << i;
      ++added;
    } else {
      infeasCheck = true;
      BOOST_LOG_TRIVIAL(trace)
          << "Algorithms::innerApproximation::addDeviatedPolyhedron: NO "
             "polyhedron added for player "
          << i;
    }
  }
  return added;
}