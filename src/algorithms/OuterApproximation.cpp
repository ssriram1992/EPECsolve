#include "algorithms/outerapproximation.h"

#include <boost/log/trivial.hpp>
#include <chrono>
#include <gurobi_c++.h>
#include <set>
#include <string>

void Algorithms::OuterApproximation::solve() {
  /**
   * Given the referenced EPEC instance, this method solves it through the outer
   * approximation Algorithm.
   */
  // Set the initial point for all countries as 0 and solve the respective LCPs?
  this->EPECObject->SolutionX.zeros(this->EPECObject->NumVariables);
  bool solved = {false};

  this->EPECObject->Stats.NumIterations = 0;
  if (this->EPECObject->Stats.AlgorithmParam.TimeLimit > 0)
    this->EPECObject->InitTime = std::chrono::high_resolution_clock::now();

  // Initialize Trees
  this->Trees = std::vector<OuterTree *>(this->EPECObject->NumPlayers, 0);
  this->incumbent =
      std::vector<OuterTree::Node *>(this->EPECObject->NumPlayers, 0);
  for (unsigned int i = 0; i < this->EPECObject->NumPlayers; i++) {
    Trees.at(i) = new OuterTree(this->outerLCP.at(i)->getNumRows());
    incumbent.at(i) = Trees.at(i)->getRoot();
    BOOST_LOG_TRIVIAL(warning) << this->outerLCP.at(i)->getNumRows();
  }

  bool rightFeas, leftFeas;
  int p = 0, comp = 0;
  std::vector<int> branchingLocations;
  std::vector<long int> branches;
  while (!solved) {
    branchingLocations.clear();
    ++this->EPECObject->Stats.NumIterations;
    BOOST_LOG_TRIVIAL(info)
        << "Algorithms::OuterApproximation::solve: Iteration "
        << std::to_string(this->EPECObject->Stats.NumIterations);

    p = 0;
    comp = 0;
    while (p < this->EPECObject->NumPlayers) {
      // Branching: location is given by this->getNextBranchLocation, and the
      // incumbent is stored in the incumbent vector
      if (incumbent.at(p)->getCumulativeBranches() ==
          Trees.at(p)->getEncodingSize())
        comp++;
      else {
        branchingLocations = this->getNextBranchLocation(p, incumbent.at(p));
        if (this->EPECObject->Stats.NumIterations == 1) {
          if (*std::max_element(branchingLocations.begin(),
                                branchingLocations.end()) < 0) {
            BOOST_LOG_TRIVIAL(info)
                << "Algorithms::OuterApproximation::solve: Player " << p
                << " has an infeasible problem.";
            this->EPECObject->Stats.Status =
                Game::EPECsolveStatus::NashEqNotFound;
            solved = true;
            return;
          }
        }
        if (branchingLocations.at(0) > -1 && branchingLocations.at(1) > -1 &&
            branchingLocations.at(0) != branchingLocations.at(1)) {
          branchingLocations.pop_back();
          branches =
              Trees.at(p)->multipleBranch(branchingLocations, *incumbent.at(p));
          BOOST_LOG_TRIVIAL(info)
              << "Algorithms::OuterApproximation::solve: Multiple branching on "
              << branchingLocations.at(0) << " and "
              << branchingLocations.at(1);
        } else {
          if (branchingLocations.at(0) > -1) {
            BOOST_LOG_TRIVIAL(info)
                << "Algorithms::OuterApproximation::solve: Player " << p
                << " - branching (Infeasible)  on " << branchingLocations.at(0);
            branches = Trees.at(p)->singleBranch(branchingLocations.at(0),
                                                 *incumbent.at(p));
            branchingLocations = {branchingLocations.at(0)};
          } else if (branchingLocations.at(1) > -1) {
            BOOST_LOG_TRIVIAL(info)
                << "Algorithms::OuterApproximation::solve: Player " << p
                << " - branching (Deviation)  on " << branchingLocations.at(1);
            branches = Trees.at(p)->singleBranch(branchingLocations.at(1),
                                                 *incumbent.at(p));
            branchingLocations = {branchingLocations.at(1)};
          } else if (branchingLocations.at(0) < 0 &&
                     branchingLocations.at(1) < 0) {
            BOOST_LOG_TRIVIAL(info)
                << "Algorithms::OuterApproximation::solve: Player " << p
                << " - branching (FirstAvailable)  on "
                << branchingLocations.at(2);
            branches = Trees.at(p)->singleBranch(branchingLocations.at(2),
                                                 *incumbent.at(p));
            branchingLocations = {branchingLocations.at(2)};
          } else {
            BOOST_LOG_TRIVIAL(info)
                << "Algorithms::OuterApproximation::solve: Player " << p
                << " - cannot branch on any variable";
          }
        }

        if (branches.at(0) < 0) {
          BOOST_LOG_TRIVIAL(error)
              << "Algorithms::OuterApproximation::solve: Player " << p
              << " - branching  on -- cannot branch.";
          throw;
        }
        // Check if any of the two children is infeasible
        auto childEncoding =
            this->Trees.at(p)->getNodes()->at(branches.at(0)).getEncoding();
        this->outerLCP.at(p)->outerApproximate(childEncoding, true);
        if (!this->outerLCP.at(p)->getFeasApprox()) {
          // This child is infeasible
          Trees.at(p)->denyBranchingLocations(*incumbent.at(p),
                                              branchingLocations);
          p--;
        } else {
          incumbent.at(p) =
              &(this->Trees.at(p)->getNodes()->at(branches.at(0)));
        }
      }

      // Increment counter
      ++p;
    }
    if (comp == this->EPECObject->NumPlayers) {
      BOOST_LOG_TRIVIAL(info) << "Algorithms::OuterApproximation::solve: "
                                 "Solved without any equilibrium.";
      this->EPECObject->Stats.Status = Game::EPECsolveStatus::NashEqNotFound;
      solved = true;
      break;
    }
    this->printCurrentApprox();
    this->EPECObject->makePlayersQPs();
    this->EPECObject->computeNashEq(
        this->EPECObject->Stats.AlgorithmParam.PureNashEquilibrium);
    if (this->EPECObject->isSolved()) {
      this->EPECObject->Stats.Status = Game::EPECsolveStatus::NashEqFound;
      BOOST_LOG_TRIVIAL(info)
          << "Algorithms::OuterApproximation::solve: Solved";
      break;
    }
  }
}

int Algorithms::OuterApproximation::getFirstBranchLocation(
    const unsigned int player, const OuterTree::Node *node) {
  /**
   * Given @p player -- containing the id of the player, returns the branching
   * decision for that node, with no complementarity condition enforced. In
   * particular, the method return the (positive) id of the complementarity
   * equation if there is a feasible branching decision at @p node, and a
   * negative value otherwise.
   * @return a positive int with the id of the complementarity to branch on, or
   * a negative value if none exists.
   */
  auto model = this->outerLCP.at(player)->LCPasMIP(true);
  unsigned int nR = this->outerLCP.at(player)->getNumRows();
  int pos = -nR;
  arma::vec z, x;
  if (this->outerLCP.at(player)->extractSols(
          model.get(), z, x, true)) // If already infeasible, nothing to branch!
  {
    std::vector<short int> v1 = this->outerLCP.at(player)->solEncode(z, x);

    double maxvalx{-1}, maxvalz{-1};
    unsigned int maxposx{0}, maxposz{0};
    for (unsigned int i = 0; i < nR; i++) {
      unsigned int varPos =
          i >= this->outerLCP.at(player)->getLStart()
              ? i + this->outerLCP.at(player)->getNumberLeader()
              : i;
      if (x(varPos) > maxvalx && node->getAllowedBranchings().at(i)) {
        maxvalx = x(varPos);
        maxposx = i;
      }
      if (z(i) > maxvalz && node->getAllowedBranchings().at(i)) {
        maxvalz = z(i);
        maxposz = i;
      }
    }
    pos = maxvalz > maxvalx ? maxposz : maxposx;
  } else {
    BOOST_LOG_TRIVIAL(debug) << "The problem is infeasible";
  }
  return pos;
}

std::vector<int> Algorithms::OuterApproximation::getNextBranchLocation(
    const unsigned int player, const OuterTree::Node *node) {
  /**
   * Given @p player -- containing the id of the player -- and @p node
   * containing a node, returns the branching decision for that node, with
   * respect to the current node. In particular, the method return the
   * (positive) id of the complementarity equation if there is a feasible
   * branching decision at @p node, and a negative value otherwise.
   * @return a positive int with the id of the complementarity to branch on, or
   * a negative value if none exists.
   */
  std::vector<int> decisions = {-1, -1, -1};
  if (this->EPECObject->NashEquilibrium) {
    // There exists a Nash Equilibrium for the outer approximation, which is not
    // a Nash Equilibrium for the game
    arma::vec x, z;
    this->EPECObject->getXWithoutHull(this->EPECObject->SolutionX, player, x);
    z = this->outerLCP.at(player)->zFromX(x);
    std::vector<short int> currentSolution =
        this->outerLCP.at(player)->solEncode(x);

    double maxInfeas = 0;

    //"The most infeasible" branching
    for (unsigned int i = 0; i < currentSolution.size(); i++) {
      unsigned int varPos =
          i >= this->outerLCP.at(player)->getLStart()
              ? i + this->outerLCP.at(player)->getNumberLeader()
              : i;
      if (x(varPos) > 0 && z(i) > 0 && node->getAllowedBranchings().at(i) &&
          currentSolution.at(i) == 0) {
        if ((x(varPos) + z(i)) > maxInfeas) {
          maxInfeas = x(varPos) + z(i);
          decisions.at(0) = i;
        }
      }
    }

    arma::vec dev;
    this->EPECObject->respondSol(dev, player, this->EPECObject->SolutionX);
    auto encoding = this->outerLCP.at(player)->solEncode(dev);

    for (unsigned int i = 0; i < encoding.size(); i++) {
      if (encoding.at(i) > 0 && node->getAllowedBranchings().at(i) &&
          currentSolution.at(i) == 0) {
        decisions.at(1) = i;
      }
    }
  }

  if (decisions.at(0) < 0 && decisions.at(1) < 0) {
    BOOST_LOG_TRIVIAL(info)
        << "Player " << player
        << ": branching with FirstBranchLocation is the only feasible choice.";
    decisions.at(2) = this->getFirstBranchLocation(player, node);
  } else if (decisions.at(1) >= 0) {
    BOOST_LOG_TRIVIAL(info)
        << "Player " << player
        << ": branching with DeviationBranching is feasible";
  } else if (decisions.at(0) >= 0) {
    BOOST_LOG_TRIVIAL(info)
        << "Player " << player
        << ": branching with MostInfeasibleBranching is feasible";
  }
  return decisions;
}

void Algorithms::OuterApproximation::printCurrentApprox() {
  /**
   * Returns a log message containing the encoding at the current outer
   * approximation iteration
   */
  BOOST_LOG_TRIVIAL(info) << "Current Node Approximation:";
  for (unsigned int p = 0; p < this->EPECObject->NumPlayers; ++p) {
    std::stringstream msg;
    msg << "\tPlayer " << p << ":";
    for (unsigned int i = 0; i < this->incumbent.at(p)->getEncoding().size();
         i++) {
      msg << "\t" << this->incumbent.at(p)->getEncoding().at(i);
    }
    BOOST_LOG_TRIVIAL(info) << msg.str();
  }
}

OuterTree::Node::Node(Node &parent, unsigned int idComp, unsigned long int id) {
  /**
   * Given the parent node address @param parent, the @param idComp to branch
   * on, and the @param id, creates a new node
   */
  this->IdComps = std::vector<unsigned int>{idComp};
  this->Encoding = parent.Encoding;
  this->Encoding.at(idComp) = true;
  this->AllowedBranchings = parent.AllowedBranchings;
  this->AllowedBranchings.at(idComp) = false;
  this->Id = id;
  this->Parent = &parent;
}

OuterTree::Node::Node(unsigned int encSize) {
  /**
   * Constructor for the root node, given the encoding size, namely the number
   * of complementarity equations
   */
  this->Encoding = std::vector<bool>(encSize, 0);
  this->Id = 0;
  this->AllowedBranchings = std::vector<bool>(encSize, true);
}

void OuterTree::denyBranchingLocation(OuterTree::Node &node,
                                      const unsigned int &location) {
  /**
   * If a complementarity equation @param location  has proven to be infeasible
   * or it isn't a candidate for branching, this method prevents any further
   * branching on it for the node @param node.
   */
  if (location >= this->EncodingSize)
    throw "OuteTree::branch idComp is larger than the encoding size.";
  if (!node.AllowedBranchings.at(location))
    BOOST_LOG_TRIVIAL(warning) << "OuterTree::denyBranchingLocation: location "
                                  "has been already denied.";
  node.AllowedBranchings.at(location) = false;
}

void OuterTree::denyBranchingLocations(OuterTree::Node &node,
                                       const std::vector<int> &locations) {
  /**
   * If a complementarity equation @param location  has proven to be infeasible
   * or it isn't a candidate for branching, this method prevents any further
   * branching on it for the node @param node.
   */
  for (auto &location : locations) {
    if (location < 0)
      throw "OuterTree::denyBranchingLocations a location is negative.";
    this->denyBranchingLocation(node, location);
  }
}

std::vector<long int> OuterTree::singleBranch(const unsigned int idComp,
                                              OuterTree::Node &t) {
  /**
   * Given the @param idComp and the parent node @param t, creates a single
   * child by branching on @param idComp.
   */
  if (idComp >= this->EncodingSize)
    throw "OuterTree::branch idComp is larger than the encoding size.";
  if (t.Encoding.at(idComp) != 0) {
    BOOST_LOG_TRIVIAL(warning)
        << "OuterTree: cannot branch on this complementary, since it already "
           "has been processed.";
    return std::vector<long int>{-1};
  }
  auto child = Node(t, idComp, this->nextIdentifier());

  this->Nodes.push_back(child);
  return std::vector<long int>{this->NodeCounter - 1};
}

std::vector<long int> OuterTree::multipleBranch(const std::vector<int> idsComp,
                                                Node &t) {
  /**
   * Given the @param idComp and the parent node @param t, creates a single
   * child by branching on @param idComp.
   */
  for (auto &idComp : idsComp) {
    if (idComp >= this->EncodingSize)
      throw "Tree::branch idComp is larger than the encoding size.";
    if (t.Encoding.at(idComp) != 0) {
      BOOST_LOG_TRIVIAL(warning)
          << "Tree: cannot branch on this complementary, since it already has "
             "been processed.";
      return std::vector<long int>{-1};
    }
  }
  auto child = Node(t, idsComp, this->nextIdentifier());

  this->Nodes.push_back(child);
  return std::vector<long int>{this->NodeCounter - 1};
}

OuterTree::Node::Node(Node &parent, std::vector<int> idsComp,
                      unsigned long int id) {
  /**
   * Given the parent node address @param parent, the @param idsComp to branch
   * on (containing all the complementarities ids), and the @param id, creates a
   * new node
   */
  this->IdComps = std::vector<unsigned int>();
  this->Encoding = parent.Encoding;
  this->AllowedBranchings = parent.AllowedBranchings;
  for (auto &idComp : idsComp) {
    if (idComp < 0)
      throw "OuterTree::Node::Node  idComp is negative.";
    this->Encoding.at(idComp) = true;
    this->AllowedBranchings.at(idComp) = false;
    this->IdComps.push_back(idComp);
  }
  this->Id = id;
  this->Parent = &parent;
}
