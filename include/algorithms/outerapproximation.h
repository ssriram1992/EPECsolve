#pragma once

#include "algorithms/algorithms.h"
#include "epecsolve.h"
#include <armadillo>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>
#include <set>
#include <string>

class OuterTree {
public:
  struct Node {
  public:
    friend class OuterTree;

    Node(unsigned int encSize);

    Node(Node &parent, unsigned int idComp, unsigned long int id);
    Node(Node &parent, std::vector<int> idComps, unsigned long int id);

    inline std::vector<unsigned int> getIdComps() const {
      return this->IdComps;
    } ///< Getter method for idComp
    inline unsigned long int getId() const {
      return this->Id;
    } ///< Getter method for id
    inline unsigned long int getCumulativeBranches() const {
      return std::count(this->AllowedBranchings.begin(),
                        this->AllowedBranchings.end(), false);
    } ///< Returns the number of variables that cannot be candidate for the
      ///< branching decisions, namely the ones on which a branching decision
      ///< has already been taken, or for which the resulting child node is
      ///< infeasible.
    inline std::vector<bool> getEncoding() const {
      return this->Encoding;
    } ///< Getter method for the encoding.

    inline std::vector<bool> getAllowedBranchings() const {
      return this->AllowedBranchings;
    } ///< Getter method for the allowed branchings

    inline Node *getParent() const {
      return this->Parent;
    } ///< Getter method for the parent node

  private:
    std::vector<unsigned int>
        IdComps; ///< Contains the branching decisions taken at the node
    std::vector<bool>
        Encoding; ///< An encoding of bool. True if the complementarity
                  ///< condition is included in the current node outer
                  ///< approximation, false otherwise.
    std::vector<bool>
        AllowedBranchings; ///< A vector where true means that the corresponding
                           ///< complementarity is a candidate for banching at
                           ///< the current node
    unsigned long int
        Id;       ///< A long int giving the numerical identifier for the node
    Node *Parent; ///< A pointer to the parent node.
  };

private:
  Node Root = Node(0);           ///< The root node of the tree
  unsigned int EncodingSize = 0; ///< The size of the encoding, namely the
                                 ///< number of complementarity equations
  unsigned int NodeCounter = 1;  ///< The counter for node ids
  std::vector<Node> Nodes{};     ///< Storage of nodes in the tree

  unsigned int nextIdentifier() {
    this->NodeCounter++;
    return (this->NodeCounter - 1);
  } ///< Increments the node counter and get the id of the new node.

public:
  explicit OuterTree(unsigned int encSize) {
    this->Root = Node(encSize);
    this->EncodingSize = encSize;
    this->Nodes.push_back(this->Root);
  } ///< Constructor of the Tree given the encoding size

  const unsigned int getEncodingSize() {
    return this->EncodingSize;
  } ///< Getter for the encoding size

  inline Node *const getRoot() {
    return &this->Root;
  } ///< Getter for the root node

  inline std::vector<Node> *getNodes() { return &this->Nodes; };

  void denyBranchingLocation(Node &node, const unsigned int &location);
  void denyBranchingLocations(Node &node, const std::vector<int> &locations);

  std::vector<long int> singleBranch(const unsigned int idComp, Node &t);

  std::vector<long int> multipleBranch(const std::vector<int> idsComp, Node &t);
};

namespace Algorithms {
///@brief This class is responsible for the outer approximation Algorithm
class OuterApproximation {
private:
  GRBEnv *env;            ///< Stores the pointer to the Gurobi Environment
  Game::EPEC *EPECObject; ///< Stores the pointer to the calling EPEC object
  std::vector<std::shared_ptr<Game::OuterLCP>> outerLCP{};
  std::vector<OuterTree *> Trees;
  std::vector<OuterTree::Node *> incumbent;

  std::vector<int> getNextBranchLocation(const unsigned int player,
                                         const OuterTree::Node *node);
  int getFirstBranchLocation(const unsigned int player,
                             const OuterTree::Node *node);

public:
  friend class EPEC;

  OuterApproximation(GRBEnv *env, Game::EPEC *EpecObj)
      : env{env}, EPECObject{EpecObj} {
    /*
     *  The constructor re-builds the LCP fields in the EPEC object as new
     * OuterLCP objects
     */
    this->EPECObject->Stats.AlgorithmParam.PolyLcp = false;
    this->outerLCP =
        std::vector<std::shared_ptr<Game::OuterLCP>>(EPECObject->NumPlayers);
    for (unsigned int i = 0; i < EPECObject->NumPlayers; i++) {
      this->outerLCP.at(i) = std::shared_ptr<Game::OuterLCP>(new Game::OuterLCP(
          this->env, *EPECObject->PlayersLowerLevels.at(i).get()));
      EPECObject->PlayersLCP.at(i) = this->outerLCP.at(i);
    }

  }; ///< Constructor requires a pointer to the Gurobi
  ///< Environment and the calling EPEC object
  void solve();
  void printCurrentApprox();
};
} // namespace Algorithms