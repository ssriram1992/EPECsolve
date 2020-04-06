#include "algorithms/outerapproximation.h"

#include <boost/log/trivial.hpp>
#include <chrono>
#include <gurobi_c++.h>
#include <set>
#include <string>

using namespace std;

void Algorithms::OuterApproximation::solve () {
	/**
	 * Given the referenced EPEC instance, this method solves it through the outer
	 * approximation Algorithm.
	 */
	// Set the initial point for all countries as 0 and solve the respective LCPs?
	this->EPECObject->SolutionX.zeros (this->EPECObject->NumVariables);
	bool solved = {false};
	bool addRand{false};
	bool infeasCheck{false};

	this->EPECObject->Stats.NumIterations = 0;
	if (this->EPECObject->Stats.AlgorithmParam.TimeLimit > 0)
		this->EPECObject->InitTime = std::chrono::high_resolution_clock::now ();

	// Initialize Trees
	this->Trees = std::vector<Tree *> (this->EPECObject->NumPlayers, 0);
	std::vector<Tree::TreeNode *> incumbent (this->EPECObject->NumPlayers, 0);
	for (unsigned int i = 0; i < this->EPECObject->NumPlayers; i++) {
		Trees.at (i) = new Tree (this->EPECObject->PlayersLCP.at (i)->getNumRows ());
		incumbent.at (i) = Trees.at (i)->getRoot ();
	}


	bool rightFeas, leftFeas;
	int p=0;
	while (!solved) {
		++this->EPECObject->Stats.NumIterations;
		BOOST_LOG_TRIVIAL(info) << "Algorithms::OuterApproximation::solve: Iteration "
		                        << to_string (this->EPECObject->Stats.NumIterations);

		while (p < this->EPECObject->NumPlayers) {
			//Branching: location is given by this->getNextBranchLocation, and the incumbent is stored in the incumbent vector
			unsigned int branchingLocation = this->getNextBranchLocation (p,
			                                                              Trees.at (p)->getEncoding (incumbent.at (p)));
			if (branchingLocation >= 0) {
				BOOST_LOG_TRIVIAL(info) << "Algorithms::OuterApproximation::solve: Player " << p << " - branching  on "
				                        << branchingLocation;
				auto leaves = Trees.at (p)->branch (branchingLocation, incumbent.at (p));
				//Check if any of the two children is infeasible

				//Right branch
				rightFeas = this->outerLCP.at (p)->checkComponentFeas (leaves.at (leaves.size () - 1)->encoding);
				//Left Branch
				leftFeas = this->outerLCP.at (p)->checkComponentFeas (leaves.at (leaves.size () - 2)->encoding);

				if (rightFeas && !leftFeas) {
					incumbent.at (p) = leaves.at (leaves.size () - 1);
					delete leaves.at (leaves.size () - 2);
				}
				else if (!rightFeas && leftFeas) {
					incumbent.at (p) = leaves.at (leaves.size () - 2);
					delete leaves.at (leaves.size () - 1);
				}
				else if (rightFeas && leftFeas) {
					//Hardcoded variable selection. Always go to the left branch, namely the one with z=0
					incumbent.at (p) = leaves.at (leaves.size () - 2);
				}
				else if (!rightFeas && !leftFeas){
					//Delete leaves and backtrack
					delete leaves.at (leaves.size () - 1);
					delete leaves.at (leaves.size () - 2);
					p--;
				}

			} else {
				BOOST_LOG_TRIVIAL(info) << "Algorithms::OuterApproximation::solve: Player " << p
				                        << " - cannot branch on any variable";
			}
			//Increment counter
			++p;
		}

	}
}

int Algorithms::OuterApproximation::getNextBranchLocation (const unsigned int player, vector<short int> *Encoding) {
	auto model = this->outerLCP.at (player)->LCPasMIP (*Encoding, true);
	int pos;
	pos = -1;
	arma::vec z, x;
	if (this->outerLCP.at (player)->extractSols (model.get (), z, x, true)) // If already infeasible, nothing to branch!
	{
		vector<short int> v1 = this->outerLCP.at (player)->solEncode (z, x);

		//this->AllPolyhedra->push_back (v1);
		//this->FixToPolies (v1);

		////////////////////
		// BRANCHING RULE //
		////////////////////
		// Branch at a large positive value
		double maxvalx{0};
		unsigned int nR = this->outerLCP.at (player)->getNumRows ();
		unsigned int maxposx{nR};
		double maxvalz{0};
		unsigned int maxposz{nR};
		for (unsigned int i = 0; i < nR; i++) {
			unsigned int varPos =
					i >= this->outerLCP.at (player)->getLStart () ? i + this->outerLCP.at (player)->getNumberLeader ()
					                                              : i;
			if (x (varPos) > maxvalx && Encoding->at (i) == 0) // If already fixed, it makes no sense!
			{
				maxvalx = x (varPos);
				maxposx = (i == 0) ? -nR : -i; // Negative of 0 is -nR by convention
			}
			if (z (i) > maxvalz && Encoding->at (i) == 0) // If already fixed, it makes no sense!
			{
				maxvalz = z (i);
				maxposz = i;
			}
		}
		pos = maxvalz > maxvalx ? maxposz : maxposx;
		///////////////////////////
		// END OF BRANCHING RULE //
		///////////////////////////
	} else {
		BOOST_LOG_TRIVIAL(debug) << "Infeasible branch";
	}
	return pos;

}
