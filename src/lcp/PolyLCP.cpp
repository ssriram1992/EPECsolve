#include "lcp/polylcp.h"
#include <algorithm>
#include <armadillo>
#include <boost/log/trivial.hpp>
#include <gurobi_c++.h>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>


bool operator==(std::vector<short int> encoding1, std::vector<short int> encoding2)
/**
 * @brief Checks if two vector<int> are of same size and hold same values in the
 * same order
 * @warning Might be deprecated, as it pollutes global namespaces
 * @returns @p true if encoding1 and encoding2 have the same elements else @p
 * false
 */
{
	if (encoding1.size() != encoding2.size())
		return false;
	for (unsigned int i = 0; i < encoding1.size(); i++) {
		if (encoding1.at(i) != encoding2.at(i))
			return false;
	}
	return true;
}

bool operator<(std::vector<short int> encoding1, std::vector<short int> encoding2)
/**
 * @details \b GrandParent:
 *  	Either the same value as the grand child, or has 0 in that location
 *
 *  \b Grandchild:
 *  	Same val as grand parent in every location, except anumVariablesY val
 * allowed, if grandparent is 0
 * @warning Might be deprecated, as it pollutes global namespaces
 * @returns @p true if encoding1 is (grand) child of encoding2
 */
{
	if (encoding1.size() != encoding2.size())
		return false;
	for (unsigned int i = 0; i < encoding1.size(); i++) {
		if (encoding1.at(i) != encoding2.at(i) &&
		    encoding1.at(i) * encoding2.at(i) != 0) {
			return false; // encoding1 is not a child of encoding2
		}
	}
	return true; // encoding1 is a child of encoding2
}

bool operator>(std::vector<int> encoding1, std::vector<int> encoding2) {
	return (encoding2 < encoding1);
}

unsigned int Game::PolyLCP::convexHull(
		arma::sp_mat &A, ///< Convex hull inequality description
		///< LHS to be stored here
		arma::vec &b)    ///< Convex hull inequality description RHS
///< to be stored here
/**
 * Computes the convex hull of the feasible region of the LCP
 */
{
	const std::vector<arma::sp_mat *> tempAi = [](spmat_Vec &uv) {
		std::vector<arma::sp_mat *> v{};
		for (const auto &x : uv)
			v.push_back(x.get());
		return v;
	}(*this->Ai);
	const std::vector<arma::vec *> tempbi = [](vec_Vec &uv) {
		std::vector<arma::vec *> v{};
		std::for_each(uv.begin(), uv.end(),
		              [&v](const std::unique_ptr<arma::vec> &ptr) {
			              v.push_back(ptr.get());
		              });
		return v;
	}(*this->bi);
	arma::sp_mat A_common;
	A_common = arma::join_cols(this->_A, -this->M);
	arma::vec b_common = arma::join_cols(this->_b, this->q);
	if (Ai->size() == 1) {
		A.zeros(Ai->at(0)->n_rows + A_common.n_rows,
		        Ai->at(0)->n_cols + A_common.n_cols);
		b.zeros(bi->at(0)->n_rows + b_common.n_rows);
		A = arma::join_cols(*Ai->at(0), A_common);
		b = arma::join_cols(*bi->at(0), b_common);
		return 1;
	} else
		return Game::convexHull(&tempAi, &tempbi, A, b, A_common, b_common);
}
Game::PolyLCP &Game::PolyLCP::addPolyFromX(const arma::vec &x, bool &ret)
/**
 * Given a <i> feasible </i> point @p x, checks if anumVariablesY polyhedron
 * that contains
 * @p x is already a part of this->Ai and this-> bi. If it is, then this does
 * nothing, except for printing a log message. If not, it adds a polyhedron
 * containing this vector.
 */
{
	const auto numCompl = this->Compl.size();
	auto encoding = this->solEncode(x);
	std::stringstream encStr;
	for (auto vv : encoding)
		encStr << vv << " ";
	BOOST_LOG_TRIVIAL(trace)
		<< "Game::PolyLCP::addPolyFromX: Handling deviation with encoding: "
		<< encStr.str() << '\n';
	// Check if the encoding polyhedron is already in this->AllPolyhedra
	for (const auto &i : AllPolyhedra) {
		std::vector<short int> bin = Utils::numToVec(i, numCompl);
		if (encoding < bin) {
			BOOST_LOG_TRIVIAL(trace) << "Game::PolyLCP::addPolyFromX: Encoding " << i
			                         << " already in All Polyhedra! ";
			ret = false;
			return *this;
		}
	}

	BOOST_LOG_TRIVIAL(trace)
		<< "Game::PolyLCP::addPolyFromX: New encoding not in All Polyhedra! ";
	// If it is not in AllPolyhedra
	// First change anumVariablesY zero indices of encoding to 1
	for (short &i : encoding) {
		if (i == 0)
			++i;
	}
	// And then add the relevant polyhedron
	ret = this->addPolyFromEncoding(encoding, false);
	// ret = true;
	return *this;
}

bool Game::PolyLCP::addPolyFromEncoding(
		const std::vector<short int>
		encoding, ///< A vector of +1 and -1 referring to which
		///< equations and variables are taking 0 value.
		bool checkFeas, ///< The polyhedron is added after ensuring feasibility, if
		///< this is true
		bool custom, ///< Should the polyhedra be pushed into a custom vector of
		///< polyhedra as opposed to LCP::Ai and LCP::bi
		spmat_Vec *custAi, ///< If custom polyhedra vector is used, pointer to
		///< vector of LHS constraint matrix
		vec_Vec *custbi /// If custom polyhedra vector is used, pointer
		/// to vector of RHS of constraints
)
/** @brief Computes the equation of the feasibility polyhedron corresponding to
 *the given @p encoding
 *	@details The computed polyhedron is always pushed into a vector of @p
 *arma::sp_mat and @p arma::vec If @p custom is false, this is the internal
 *attribute of LCP, which are LCP::Ai and LCP::bi. Otherwise, the vectors can be
 *provided as arguments.
 *	@p true value to @p checkFeas ensures that the polyhedron is pushed @e
 *only if it is feasible.
 * @returns @p true if successfully added, else false
 *	@warning Does not entertain 0 in the elements of *encoding. Only +1/-1
 *are allowed to not encounter undefined behavior. As a result, not meant for
 *high level code. Instead use LCP::addPoliesFromEncoding.
 */
{
	unsigned int encodingNumber = Utils::vecToNum(encoding);
	BOOST_LOG_TRIVIAL(trace)
		<< "Game::PolyLCP::addPolyFromEncoding: Working on polyhedron #"
		<< encodingNumber;

	bool eval = false;
	if (checkFeas)
		eval = this->checkPolyFeas(encoding);
	else
		eval = true;

	if (eval) {
		if (!custom && !AllPolyhedra.empty()) {
			if (AllPolyhedra.find(encodingNumber) != AllPolyhedra.end()) {
				BOOST_LOG_TRIVIAL(trace) << "Game::PolyLCP::addPolyFromEncoding: "
				                            "Previously added polyhedron #"
				                         << encodingNumber;
				return false;
			}
		}
		std::unique_ptr<arma::sp_mat> Aii =
				std::unique_ptr<arma::sp_mat>(new arma::sp_mat(nR, nC));
		Aii->zeros();
		std::unique_ptr<arma::vec> bii =
				std::unique_ptr<arma::vec>(new arma::vec(nR, arma::fill::zeros));
		for (unsigned int i = 0; i < this->nR; i++) {
			if (encoding.at(i) == 0) {
				throw("Error in Game::PolyLCP::addPolyFromEncoding. 0s not allowed in "
				      "argument vector");
			}
			if (encoding.at(i) == 1) // Equation to be fixed top zero
			{
				for (auto j = this->M.begin_row(i); j != this->M.end_row(i); ++j)
					if (!this->isZero((*j)))
						Aii->at(i, j.col()) =
								(*j); // Only mess with non-zero elements of a sparse matrix!
				bii->at(i) = -this->q(i);
			} else // Variable to be fixed to zero, i.e. x(j) <= 0 constraint to be
				// added
			{
				unsigned int variablePosition =
						(i >= this->LeadStart) ? i + this->NumberLeader : i;
				Aii->at(i, variablePosition) = 1;
				bii->at(i) = 0;
			}
		}
		if (custom) {
			custAi->push_back(std::move(Aii));
			custbi->push_back(std::move(bii));
		} else {
			AllPolyhedra.insert(encodingNumber);
			this->Ai->push_back(std::move(Aii));
			this->bi->push_back(std::move(bii));
		}
		return true; // Successfully added
	}
	BOOST_LOG_TRIVIAL(trace) << "Game::PolyLCP::addPolyFromEncoding: Checkfeas + "
	                            "Infeasible polyhedron #"
	                         << encodingNumber;
	return false;
}

Game::PolyLCP &Game::PolyLCP::addPoliesFromEncoding(
		const std::vector<short int>
		encoding, ///< A vector of +1, 0 and -1 referring to which
		///< equations and variables are taking 0 value.
		bool checkFeas, ///< The polyhedron is added after ensuring feasibility, if
		///< this is true
		bool custom, ///< Should the polyhedra be pushed into a custom vector of
		///< polyhedra as opposed to LCP::Ai and LCP::bi
		spmat_Vec *custAi, ///< If custom polyhedra vector is used, pointer to
		///< vector of LHS constraint matrix
		vec_Vec *custbi /// If custom polyhedra vector is used, pointer
		/// to vector of RHS of constraints
)
/** @brief Computes the equation of the feasibility polyhedron corresponding to
 *the given @p encoding
 *	@details The computed polyhedron are always pushed into a vector of @p
 *arma::sp_mat and @p arma::vec If @p custom is false, this is the internal
 *attribute of LCP, which are LCP::Ai and LCP::bi. Otherwise, the vectors can be
 *provided as arguments.
 *	@p true value to @p checkFeas ensures that @e each polyhedron that is
 *pushed is feasible. not meant for high level code. Instead use
 *LCP::addPoliesFromEncoding.
 *	@note A value of 0 in @p *encoding implies that polyhedron corresponding
 *to fixing the corresponding variable as well as the equation become candidates
 *to pushed into the vector. Hence this is preferred over
 *LCP::addPolyFromEncoding for high-level usage.
 */
{
	bool flag = false; // flag that there may be multiple polyhedra, i.e. 0 in
	// some encoding entry
	std::vector<short int> encodingCopy(encoding);
	unsigned int i = 0;
	for (i = 0; i < this->nR; i++) {
		if (encoding.at(i) == 0) {
			flag = true;
			break;
		}
	}
	if (flag) {
		encodingCopy[i] = 1;
		this->addPoliesFromEncoding(encodingCopy, checkFeas, custom, custAi,
		                            custbi);
		encodingCopy[i] = -1;
		this->addPoliesFromEncoding(encodingCopy, checkFeas, custom, custAi,
		                            custbi);
	} else
		this->addPolyFromEncoding(encoding, checkFeas, custom, custAi, custbi);
	return *this;
}

unsigned long int Game::PolyLCP::getNextPoly(Game::EPECAddPolyMethod method) {
	/**
	 * Returns a polyhedron (in its decimal encoding) that is neither already
	 * known to be infeasible, nor already added in the inner approximation
	 * representation.
	 */

	switch (method) {
		case Game::EPECAddPolyMethod::Sequential: {
			while (this->SequentialPolyCounter < this->MaxTheoreticalPoly) {
				const auto isAll =
						AllPolyhedra.find(this->SequentialPolyCounter) != AllPolyhedra.end();
				const auto isInfeas = InfeasiblePoly.find(this->SequentialPolyCounter) !=
				                      InfeasiblePoly.end();
				this->SequentialPolyCounter++;
				if (!isAll && !isInfeas) {
					return this->SequentialPolyCounter - 1;
				}
			}
			return this->MaxTheoreticalPoly;
		} break;
		case Game::EPECAddPolyMethod::ReverseSequential: {
			while (this->ReverseSequentialPolyCounter >= 0) {
				const auto isAll =
						AllPolyhedra.find(this->ReverseSequentialPolyCounter) !=
						AllPolyhedra.end();
				const auto isInfeas =
						InfeasiblePoly.find(this->ReverseSequentialPolyCounter) !=
						InfeasiblePoly.end();
				this->ReverseSequentialPolyCounter--;
				if (!isAll && !isInfeas) {
					return this->ReverseSequentialPolyCounter + 1;
				}
			}
			return this->MaxTheoreticalPoly;
		} break;
		case Game::EPECAddPolyMethod::Random: {
			static std::mt19937 engine{this->AddPolyMethodSeed};
			std::uniform_int_distribution<unsigned long int> dist(
					0, this->MaxTheoreticalPoly - 1);
			if ((InfeasiblePoly.size() + AllPolyhedra.size()) ==
			    this->MaxTheoreticalPoly)
				return this->MaxTheoreticalPoly;
			while (true) {
				auto randomPolyId = dist(engine);
				const auto isAll = AllPolyhedra.find(randomPolyId) != AllPolyhedra.end();
				const auto isInfeas =
						InfeasiblePoly.find(randomPolyId) != InfeasiblePoly.end();
				if (!isAll && !isInfeas)
					return randomPolyId;
			}
		}
	}
}

std::set<std::vector<short int>>
Game::PolyLCP::addAPoly(unsigned long int nPoly, Game::EPECAddPolyMethod method,
                        std::set<std::vector<short int>> polyhedra) {
	/**
	 * Tries to add at most @p nPoly number of polyhedra to the inner
	 * approximation representation of the current LCP. The set of added polyhedra
	 * (+1/-1 encoding) is appended to  @p polyhedra and returned. The only reason
	 * fewer polyhedra might be added is that the fewer polyhedra already
	 * represent the feasible region of the LCP.
	 * @p method is casted from Game::EPEC::EPECAddPolyMethod
	 */

	// We already have polyhedra AllPolyhedra and in
	// InfeasiblePoly, that are known to be infeasible.
	// Effective maximum of number of polyhedra that can be added
	// at most
	const auto numCompl = this->Compl.size();

	if (this->MaxTheoreticalPoly <
	    nPoly) {                 // If you cannot add that numVariablesY polyhedra
		BOOST_LOG_TRIVIAL(warning) // Then issue a warning
					<< "Warning in Game::PolyLCP::randomPoly: "
					<< "Cannot add " << nPoly << " polyhedra. Promising a maximum of "
					<< this->MaxTheoreticalPoly;
		nPoly = this->MaxTheoreticalPoly; // and update maximum possibly addable
	}

	if (nPoly == 0) // If nothing to be added, then nothing to be done
		return polyhedra;

	if (nPoly < 0) // There is no way that this can happen!
	{
		BOOST_LOG_TRIVIAL(error) << "nPoly can't be negative, i.e., " << nPoly;
		throw("Error in Game::PolyLCP::addAPoly: nPoly reached a negative value!");
	}

	bool complete{false};
	while (!complete) {
		auto choiceDecimal = this->getNextPoly(method);
		if (choiceDecimal >= this->MaxTheoreticalPoly)
			return polyhedra;

		const std::vector<short int> choice = Utils::numToVec(choiceDecimal, numCompl);
		auto added = this->addPolyFromEncoding(choice, true);
		if (added) // If choice is added to All Polyhedra
		{
			polyhedra.insert(choice); // Add it to set of added polyhedra
			if (polyhedra.size() == nPoly) {
				return polyhedra;
			}
		}
	}
	return polyhedra;
}
bool Game::PolyLCP::addThePoly(const unsigned long int &decimalEncoding) {
	if (this->MaxTheoreticalPoly < decimalEncoding) {
		// This polyhedron does not exist
		BOOST_LOG_TRIVIAL(warning)
			<< "Warning in Game::PolyLCP::addThePoly: Cannot add "
			<< decimalEncoding << " polyhedra, since it does not exist!";
		return false;
	}
	const unsigned int numCompl = this->Compl.size();
	const std::vector<short int> choice = Utils::numToVec(decimalEncoding, numCompl);
	return this->addPolyFromEncoding(choice, true);
}

Game::PolyLCP &Game::PolyLCP::enumerateAll(
		const bool
		solveLP ///< Should the polyhedra added be checked for feasibility?
)
/**
 * @brief Brute force computation of LCP feasible region
 * @details Computes all @f$2^n@f$ polyhedra defining the LCP feasible region.
 * Th ese are always added to LCP::Ai and LCP::bi
 */
{
	std::vector<short int> encoding = std::vector<short int>(nR, 0);
	this->Ai->clear();
	this->bi->clear();
	this->addPoliesFromEncoding(encoding, solveLP);
	if (this->Ai->empty()) {
		BOOST_LOG_TRIVIAL(warning)
			<< "Empty vector of polyhedra given! Problem might be infeasible."
			<< '\n';
		// 0 <= -1 for infeasability
		std::unique_ptr<arma::sp_mat> A(new arma::sp_mat(1, this->M.n_cols));
		std::unique_ptr<arma::vec> b(new arma::vec(1));
		b->at(0) = -1;
		this->Ai->push_back(std::move(A));
		this->bi->push_back(std::move(b));
	}
	return *this;
}

void Game::PolyLCP::makeQP(
		Game::QP_Objective
		&QP_obj, ///< The objective function of the QP to be returned. @warning
		///< Size of this parameter might change!
		Game::QP_Param &QP ///< The output parameter where the final Game::QP_Param
		///< object is stored

) {
	// Original sizes
	if (this->Ai->empty())
		return;
	const unsigned int oldNumVariablesX{
			static_cast<unsigned int>(QP_obj.C.n_cols)};

	Game::QP_Constraints QP_cons;
	this->FeasiblePolyhedra = this->convexHull(QP_cons.B, QP_cons.b);
	BOOST_LOG_TRIVIAL(trace) << "PolyLCP::makeQP: No. feasible polyhedra: "
	                         << this->FeasiblePolyhedra;
	// Updated size after convex hull has been computed.
	const unsigned int numConstraints{
			static_cast<unsigned int>(QP_cons.B.n_rows)};
	const unsigned int numVariablesY{static_cast<unsigned int>(QP_cons.B.n_cols)};
	// Resizing entities.
	QP_cons.A.zeros(numConstraints, oldNumVariablesX);
	QP_obj.c = Utils::resizePatch(QP_obj.c, numVariablesY, 1);
	QP_obj.C = Utils::resizePatch(QP_obj.C, numVariablesY, oldNumVariablesX);
	QP_obj.Q = Utils::resizePatch(QP_obj.Q, numVariablesY, numVariablesY);
	// Setting the QP_Param object
	QP.set(QP_obj, QP_cons);
}

std::string Game::PolyLCP::feasabilityDetailString() const {
	std::stringstream ss;
	ss << "\tProven feasible: ";
	for (auto vv : this->AllPolyhedra)
		ss << vv << ' ';
	// ss << "\tProven infeasible: ";
	// for (auto vv : this->InfeasiblePoly)
	// ss << vv << ' ';

	return ss.str();
}

unsigned long Game::PolyLCP::convNumPoly() const {
	/**
	 * To be used in interaction with Game::LCP::convexHull.
	 * Gives the number of polyhedra in the current inner approximation of the LCP
	 * feasible region.
	 */
	return this->AllPolyhedra.size();
}

unsigned int Game::PolyLCP::convPolyPosition(const unsigned long int i) const {
	/**
	 * For the convex hull of the LCP feasible region computed, a bunch of
	 * variables are added for extended formulation and the added variables c
	 */
	const unsigned int nPoly = this->convNumPoly();
	if (i > nPoly) {
		BOOST_LOG_TRIVIAL(error) << "Error in Game::PolyLCP::convPolyPosition: "
		                            "Invalid argument. Out of bounds for i";
		throw("Error in Game::PolyLCP::convPolyPosition: Invalid "
		      "argument. Out of bounds for i");
	}
	const unsigned int nC = this->M.n_cols;
	return nC + i * nC;
}

unsigned int Game::PolyLCP::convPolyWeight(const unsigned long int i) const {
	/**
	 * To be used in interaction with Game::LCP::convexHull.
	 * Gives the position of the variable, which assigns the convex weight to the
	 * i-th polyhedron.
	 *
	 * However, if the inner approximation has exactly one polyhedron,
	 * then returns 0.
	 */
	const unsigned int nPoly = this->convNumPoly();
	if (nPoly <= 1) {
		return 0;
	}
	if (i > nPoly) {
		throw("Error in Game::PolyLCP::convPolyWeight: "
		      "Invalid argument. Out of bounds for i");
	}
	const unsigned int nC = this->M.n_cols;

	return nC + nPoly * nC + i;
}

bool Game::PolyLCP::checkPolyFeas(
		const unsigned long int
		&decimalEncoding ///< Decimal encoding for the polyhedron
) {
	return this->checkPolyFeas(Utils::numToVec(decimalEncoding, this->Compl.size()));
}

bool Game::PolyLCP::checkPolyFeas(
		const std::vector<short int>
		&encoding ///< A vector of +1 and -1 referring to which
		///< equations and variables are taking 0 value.)
) {

	unsigned long int encodingNumber = Utils::vecToNum(encoding);

	if (InfeasiblePoly.find(encodingNumber) != InfeasiblePoly.end()) {
		BOOST_LOG_TRIVIAL(trace)
			<< "Game::PolyLCP::checkPolyFeas: Previously known "
			   "infeasible polyhedron. "
			<< encodingNumber;
		return false;
	}

	if (FeasiblePoly.find(encodingNumber) != FeasiblePoly.end()) {
		BOOST_LOG_TRIVIAL(trace)
			<< "Game::PolyLCP::checkPolyFeas: Previously known "
			   "feasible polyhedron."
			<< encodingNumber;
		return true;
	}

	unsigned int count{0};
	try {
		makeRelaxed();
		GRBModel model(this->RlxdModel);
		for (auto i : encoding) {
			if (i > 0)
				model.getVarByName("z_" + std::to_string(count)).set(GRB_DoubleAttr_UB, 0);
			if (i < 0)
				model
						.getVarByName("x_" + std::to_string(count >= this->LeadStart
						                               ? count + NumberLeader
						                               : count))
						.set(GRB_DoubleAttr_UB, 0);
			count++;
		}
		model.set(GRB_IntParam_OutputFlag, 0);
		model.optimize();
		if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
			FeasiblePoly.insert(encodingNumber);
			return true;
		} else {
			BOOST_LOG_TRIVIAL(trace)
				<< "Game::PolyLCP::checkPolyFeas: Detected infeasibility of "
				<< encodingNumber << " (GRB_STATUS=" << model.get(GRB_IntAttr_Status)
				<< ")";
			InfeasiblePoly.insert(encodingNumber);
			return false;
		}
	} catch (const char *e) {
		std::cerr << "Error in Game::PolyLCP::checkPolyFeas: " << e << '\n';
		throw;
	} catch (std::string e) {
		std::cerr << "String: Error in Game::PolyLCP::checkPolyFeas: " << e << '\n';
		throw;
	} catch (std::exception &e) {
		std::cerr << "Exception: Error in Game::PolyLCP::checkPolyFeas: " << e.what()
		     << '\n';
		throw;
	} catch (GRBException &e) {
		std::cerr << "GRBException: Error in Game::PolyLCP::checkPolyFeas: "
		     << e.getErrorCode() << ": " << e.getMessage() << '\n';
		throw;
	}
	return false;
}