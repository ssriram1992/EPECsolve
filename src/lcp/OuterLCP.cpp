#include "lcp/outerlcp.h"
#include <boost/log/trivial.hpp>

unsigned int Game::OuterLCP::convexHull (arma::sp_mat &A, ///< Convex hull inequality description
		///< LHS to be stored here
		                                 arma::vec &b) ///< Convex hull inequality description RHS
///< to be stored here
/**
 * Computes the convex hull of the feasible region of the LCP
 */
{
	const std::vector<arma::sp_mat *> tempAi = [] (spmat_Vec &uv) {
		std::vector<arma::sp_mat *> v{};
		for (const auto &x : uv)
			v.push_back (x.get ());
		return v;
	} (*this->Ai);
	const auto tempbi = [] (vec_Vec &uv) {
		std::vector<arma::vec *> v{};
		std::for_each (uv.begin (), uv.end (), [&v] (const std::unique_ptr<arma::vec> &ptr) {
			v.push_back (ptr.get ());
		});
		return v;
	} (*this->bi);
	arma::sp_mat A_common;
	A_common = arma::join_cols (this->_A, -this->M);
	arma::vec b_common = arma::join_cols (this->_b, this->q);
	if (Ai->size () == 1) {
		A.zeros (Ai->at (0)->n_rows + A_common.n_rows, Ai->at (0)->n_cols + A_common.n_cols);
		b.zeros (bi->at (0)->n_rows + b_common.n_rows);
		A = arma::join_cols (*Ai->at (0), A_common);
		b = arma::join_cols (*bi->at (0), b_common);
		return 1;
	} else
		return Game::convexHull (&tempAi, &tempbi, A, b, A_common, b_common);
}

void Game::OuterLCP::makeQP (Game::QP_Objective &QP_obj, Game::QP_Param &QP) {

	// Original sizes
	if (this->Ai->empty ())
		return;
	const unsigned int oldNumVariablesX{static_cast<unsigned int>(QP_obj.C.n_cols)};

	Game::QP_Constraints QP_cons;
	int components = this->convexHull (QP_cons.B, QP_cons.b);
	BOOST_LOG_TRIVIAL(trace) << "OuterLCP::makeQP: No. components: " << components;
	// Updated size after convex hull has been computed.
	const unsigned int numConstraints{static_cast<unsigned int>(QP_cons.B.n_rows)};
	const unsigned int oldNumVariablesY{static_cast<unsigned int>(QP_cons.B.n_cols)};
	// Resizing entities.
	QP_cons.A.zeros (numConstraints, oldNumVariablesX);
	QP_obj.c = Utils::resizePatch (QP_obj.c, oldNumVariablesY, 1);
	QP_obj.C = Utils::resizePatch (QP_obj.C, oldNumVariablesY, oldNumVariablesX);
	QP_obj.Q = Utils::resizePatch (QP_obj.Q, oldNumVariablesY, oldNumVariablesY);
	// Setting the QP_Param object
	QP.set (QP_obj, QP_cons);
}

void Game::OuterLCP::outerApproximate (const std::vector<bool> encoding, bool clear) {
	if (encoding.size () != this->Compl.size ()) {
		BOOST_LOG_TRIVIAL(error) << "Game::OuterLCP::outerApproximate: wrong encoding size";
		throw;
	}
	if (clear) {
		this->clearApproximation ();
		BOOST_LOG_TRIVIAL(error) << "Game::OuterLCP::outerApproximate: clearing current approximation.";
	}
	std::vector<short int> localEncoding = {};
	// We push 2 for each complementary that has to be fixed either to +1 or -1
	// And 0 for each one which is not processed (yet)
	for (bool i : encoding) {
		if (i)
			localEncoding.push_back (2);
		else
			localEncoding.push_back (0);
	}
	this->addChildComponents (localEncoding);
}

void Game::OuterLCP::addChildComponents (const std::vector<short int> encoding) {
	std::vector<short int> localEncoding (encoding);
	unsigned int i = 0;
	bool flag = false;
	for (i = 0; i < this->nR; i++) {
		if (encoding.at (i) == 2) {
			flag = true;
			break;
		}
	}
	if (flag) {
		localEncoding[i] = 1;
		this->addChildComponents (localEncoding);
		localEncoding[i] = -1;
		this->addChildComponents (localEncoding);
	} else
		this->addComponent (encoding, true);
}

bool Game::OuterLCP::addComponent (const std::vector<short int> encoding, ///< A vector of +1,-1 and zeros referring to which
		///< equations and variables are taking 0 value. +1 means equation set to
		///< zero, -1 variable, and zero  none of the two
		                           bool checkFeas, ///< The component is added after ensuring feasibility, if
		///< this is true
		                           bool custom, ///< Should the components be pushed into a custom vector of
		///< polyhedra as opposed to OuterLCP::Ai and OuterLCP::bi
		                           spmat_Vec *custAi, ///< If custom polyhedra vector is used, pointer to the
		///< LHS matrix
		                           vec_Vec *custbi ///< If custom polyhedra vector is used, pointer to the RHS
		///< vector
) {
	unsigned long fixNumber = Utils::vecToNum (encoding);
	BOOST_LOG_TRIVIAL(trace) << "Game::OuterLCP::addComponent: Working on polyhedron #" << fixNumber;
	bool eval;
	if (checkFeas)
		eval = this->checkComponentFeas (encoding);
	else
		eval = true;

	if (eval) {
		this->feasApprox= true;
		if (!custom && !Approximation.empty ()) {
			if (Approximation.find (fixNumber) != Approximation.end ()) {
				BOOST_LOG_TRIVIAL(trace) << "Game::OuterLCP::addComponent: Previously added polyhedron #" << fixNumber;
				return false;
			}
		}
		std::unique_ptr<arma::sp_mat> Aii = std::unique_ptr<arma::sp_mat> (new arma::sp_mat (nR, nC));
		Aii->zeros ();
		std::unique_ptr<arma::vec> bii = std::unique_ptr<arma::vec> (new arma::vec (nR, arma::fill::zeros));
		for (unsigned int i = 0; i < this->nR; i++) {
			switch (encoding.at (i)) {
				case 1: {
					for (auto j = this->M.begin_row (i); j != this->M.end_row (i); ++j)
						if (!this->isZero ((*j)))
							Aii->at (i, j.col ()) = (*j); // Only mess with non-zero elements of a sparse matrix!
					bii->at (i) = -this->q (i);
				}
					break;
				case -1: {
					unsigned int variablePosition = (i >= this->LeadStart) ? i + this->NumberLeader : i;
					Aii->at (i, variablePosition) = 1;
					bii->at (i) = 0;
				}
					break;
				case 0:
					break;
				default: {
					BOOST_LOG_TRIVIAL(error) << "Game::OuterLCP::addComponent: Non-allowed value in encoding: "
					                         << encoding.at (i);
					throw "Game::OuterLCP::addComponent: Non-allowed encoding.";
				}
			}
		}
		if (custom) {
			custAi->push_back (std::move (Aii));
			custbi->push_back (std::move (bii));
		} else {
			Approximation.insert (fixNumber);
			this->Ai->push_back (std::move (Aii));
			this->bi->push_back (std::move (bii));
		}
		return true; // Successfully added
	}
	BOOST_LOG_TRIVIAL(trace) << "Game::OuterLCP::addComponent: Checkfeas + Infeasible polyhedron #" << fixNumber;
	return false;
}

bool Game::OuterLCP::checkComponentFeas (const std::vector<short int> &encoding) {
	unsigned long int fixNumber = Utils::vecToNum (encoding);
	if (InfeasibleComponents.find (fixNumber) != InfeasibleComponents.end ()) {
		BOOST_LOG_TRIVIAL(trace) << "Game::OuterLCP::checkComponentFeas: Previously known "
		                            "infeasible component #" << fixNumber;
		return false;
	}

	if (FeasibleComponents.find (fixNumber) != FeasibleComponents.end ()) {
		BOOST_LOG_TRIVIAL(trace) << "Game::OuterLCP::checkComponentFeas: Previously known "
		                            "feasible polyhedron #" << fixNumber;
		return true;
	}
	for (auto element : InfeasibleComponents) {
		if (this->isParent (Utils::numToVec (element, this->Compl.size ()), encoding)) {
			BOOST_LOG_TRIVIAL(trace) << "Game::OuterLCP::checkComponentFeas: #" << fixNumber << " is a child "
			                                                                                    "of an infeasible polyhedron";
			return false;
		}
	}

	unsigned int count{0};
	try {
		makeRelaxed ();
		GRBModel model (this->RlxdModel);
		for (auto i : encoding) {
			if (i > 0)
				model.getVarByName ("z_" + std::to_string (count)).set (GRB_DoubleAttr_UB, 0);
			if (i < 0)
				model.getVarByName ("x_" + std::to_string (count >= this->LeadStart ? count + NumberLeader : count)).set (
						GRB_DoubleAttr_UB, 0);
			count++;
		}
		model.set (GRB_IntParam_OutputFlag, 0);
		model.optimize ();
		if (model.get (GRB_IntAttr_Status) == GRB_OPTIMAL) {
			FeasibleComponents.insert (fixNumber);
			return true;
		} else {
			BOOST_LOG_TRIVIAL(trace) << "Game::OuterLCP::checkComponentFeas: Detected infeasibility of #" << fixNumber
			                         << " (GRB_STATUS=" << model.get (GRB_IntAttr_Status) << ")";
			InfeasibleComponents.insert (fixNumber);
			return false;
		}
	} catch (const char *e) {
		std::cerr << "Error in Game::OuterLCP::checkComponentFeas: " << e << '\n';
		throw;
	} catch (std::string &e) {
		std::cerr << "String: Error in Game::OuterLCP::checkComponentFeas: " << e << '\n';
		throw;
	} catch (std::exception &e) {
		std::cerr << "Exception: Error in Game::OuterLCP::checkComponentFeas: " << e.what () << '\n';
		throw;
	} catch (GRBException &e) {
		std::cerr << "GRBException: Error in Game::OuterLCP::checkComponentFeas: " << e.getErrorCode () << ": "
		     << e.getMessage () << '\n';
		throw;
	}
	return false;
}

bool Game::OuterLCP::isParent (const
std::vector<short int> &father, const std::vector<short int> &child) {
	for (unsigned long i = 0; i < father.size (); ++i) {
		if (father.at (i) != 0) {
			if (child.at (i) != father.at (i))
				return false;
		}
	}
	return true;
}