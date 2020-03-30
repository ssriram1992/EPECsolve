#pragma once
#include "algorithms.h"
#include <armadillo>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>
#include <set>
#include <string>

class Tree {
public:
  struct TreeNode {
    unsigned int idComp;   ///< Contains the branching id
    bool brancingDirection; ///< 0 branches on var, 1 on equation
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
    left->brancingDirection = false;
    right->brancingDirection = true;
    left->encoding = t->encoding;
    left->encoding.at(idComp) = -1;
    right->encoding = t->encoding;
    right->encoding.at(idComp) = +1;
    this->nodes.push_back(left);
    this->nodes.push_back(right);
    this->nodes.erase(std::remove(this->nodes.begin(), this->nodes.end(), t), this->nodes.end());
    delete t;
    return this->nodes;
  }
};

namespace Algorithms {
///@brief This class is responsible for the outer approximation algorithm
class outerApproximation {
private:
  GRBEnv *env;      ///< Stores the pointer to the Gurobi Environment
  EPEC *EPECObject; ///< Stores the pointer to the calling EPEC object
  std::vector<Tree *> Trees;

public:
  friend class EPEC;
  outerApproximation(GRBEnv *env, EPEC *EpecObj)
      : env{env},
        EPECObject{
            EpecObj} {}; ///< Constructor requires a pointer to the Gurobi
                         ///< Environment and the calling EPEC object
  void solve();
};
} // namespace Algorithms