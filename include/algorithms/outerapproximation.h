#pragma once
#include "algorithms/algorithms.h"
#include "epecsolve.h"
#include <armadillo>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>
#include <set>
#include <string>

class Tree {
public:
  struct TreeNode {
    unsigned int idComp;     ///< Contains the branching id
    bool branchingDirection; ///< 0 branches on var, 1 on equation
    unsigned int vectorLoc;
    Col<int> encoding;
  };

private:
  TreeNode *root;
  unsigned int encodingSize = 0;
  unsigned int nodeCounter = 0;
  std::vector<TreeNode *> nodes;

  unsigned int getNextIdentifier() {
    this->nodeCounter++;
    return (this->nodeCounter - 1);
  }

public:
  explicit Tree(unsigned int encSize) {
    this->root = new TreeNode();
    this->encodingSize = encSize;
    this->root->encoding.zeros(encSize);
    this->root->vectorLoc = getNextIdentifier();
    this->nodes.push_back(this->root);
  }
  ~Tree() { delete (this); }
  TreeNode *getRoot() { return Tree::root; }
  arma::Col<int> *getEncoding(TreeNode *t) { return &t->encoding; }

  std::vector<TreeNode *> branch(unsigned int idComp, TreeNode *t) {
    if (t == nullptr)
      throw "Tree: null pointer provided.";
    auto *left = new TreeNode();
    auto *right = new TreeNode();
    left->branchingDirection = false;
    right->branchingDirection = true;
    left->encoding = t->encoding;
    left->encoding.at(idComp) = -1;
    right->encoding = t->encoding;
    right->encoding.at(idComp) = +1;
    this->nodes.push_back(left);
    this->nodes.push_back(right);
    this->nodes.erase(std::remove(this->nodes.begin(), this->nodes.end(), t),
                      this->nodes.end());
    delete t;
    return this->nodes;
  }
};

namespace Algorithms {
///@brief This class is responsible for the outer approximation Algorithm
class OuterApproximation {
private:
  GRBEnv *env;      ///< Stores the pointer to the Gurobi Environment
  EPEC *EPECObject; ///< Stores the pointer to the calling EPEC object
  std::vector<std::shared_ptr<Game::OuterLCP>> outerLCP{};
  std::vector<Tree *> Trees;

public:
  friend class EPEC;
  OuterApproximation(GRBEnv *env, EPEC *EpecObj)
      : env{env}, EPECObject{EpecObj} {

    /*
     *  The method will reassign the LCP fields in the EPEC object to new
     * PolyLCP objects
     */
    this->EPECObject->Stats.AlgorithmParam.PolyLcp = false;
    this->outerLCP =
        std::vector<std::shared_ptr<Game::OuterLCP>>(EPECObject->NumPlayers);
    for (unsigned int i = 0; i < EPECObject->NumPlayers; i++) {
      this->outerLCP.at(i) = std::shared_ptr<Game::OuterLCP>(
          new OuterLCP(this->env, *EPECObject->PlayersLowerLevels.at(i).get()));
      EPECObject->PlayersLCP.at(i) = this->outerLCP.at(i);
    }

  }; ///< Constructor requires a pointer to the Gurobi
     ///< Environment and the calling EPEC object
  void solve();
};
} // namespace Algorithms