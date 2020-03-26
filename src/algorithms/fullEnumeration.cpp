#include "fullEnumeration.h"

using namespace std;

void Algorithms::fullEnumeration::solve() {
  /** @brief Solve the referenced EPEC instance with the full enumeration
   * @p excludelist contains the set of excluded polyhedra combinations.
   */
  for (unsigned int i = 0; i < this->EPECObject->nCountr; ++i)
    this->poly_LCP.at(i)->EnumerateAll(true);
  this->EPECObject->make_country_QP();
  BOOST_LOG_TRIVIAL(trace)
    << "Algorithms::fullEnumeration::solve: Starting fullEnumeration search";
  this->EPECObject->computeNashEq(this->EPECObject->Stats.AlgorithmParam.pureNE,
                                  this->EPECObject->Stats.AlgorithmParam.timeLimit);
  if (this->EPECObject->isSolved()) {
    this->EPECObject->Stats.status = Game::EPECsolveStatus::nashEqFound;
    if (this->EPECObject->isPureStrategy())
      this->EPECObject->Stats.pureNE = true;
  }
  //Post Solving
  this->postSolving();
}