#include "FacilityData.h"
#include<iostream>
#include<memory>
#include<armadillo>
#include<array>
#include <cassert>

using namespace std;

FacilityData::FacilityData::FacilityData(arma::vec &&c, arma::mat &&c_uv, arma::vec &&D, arma::mat &&Z,
                                         arma::vec &&alpha) {
/**
 * @brief
 * Construct a FacilityData object defining both leader and followers variables
 *  @p \f$c\f$ represents the vector of unit costs for each facility
 *  @p \f$c_{uv}\f$ is the vector of transportation costs
 *  @p \f$D\f$ is the vector of demand caps for the leaders
 *  @p \f$Z\f$ is a matrix defined with @p D.size rows and @p c columns, representing the sign of the interaction between demands
 *  @p \f$alpha\f$ is the double vector containing interaction coefficients
 */
    this->initialize(c, c_uv, D, Z, alpha);
}


FacilityData::FacilityData &
FacilityData::FacilityData::initialize(arma::vec &c, arma::mat &c_uv, arma::vec &D,
                                       arma::mat &Z, arma::vec &alpha) {
    /**
 * @brief
 * Construct a FacilityData object defining both leader and followers variables
 * asserting that each object is properly shaped
 * @warning for internal use only
 */
    assert(c.size() > 0 && "c has less than 2 entries.\n");
    assert(c_uv.is_square() && "c_uv is not a square matrix.\n");
    assert(trace(c_uv) == 0 && "c_uv has non-zero elements on its diagonal.\n");
    assert(c_uv.is_symmetric() && "c_uv is not a symmetric matrix.\n");
    assert(c.size() == c_uv.n_cols && "c and c_uv sizes mismatch.\n");
    assert(D.size() == alpha.size() && "c and c_uv sizes mismatch.\n");
    assert(D.size() == Z.n_rows && "D and Z.n_rows sizes mismatch.\n");
    assert(c.size() == Z.n_cols && "c and Z.n_cols sizes mismatch.\n");
    this->num_locations = c.size();
    this->num_leaders = D.size();
    this->location_costs = (c);
    this->transporation_costs = (c_uv);
    this->D = (D);
    this->Z = (Z);
    this->alpha = (alpha);
    return *this;
}


void FacilityData::FacilityData::write(string filename) const {
    this->location_costs.save(filename + "_c.txt", arma::file_type::arma_ascii);
    this->transporation_costs.save(filename + "_c_uv.txt", arma::file_type::arma_ascii);
    this->D.save(filename + "_D.txt", arma::file_type::arma_ascii);
    this->Z.save(filename + "_Z.txt", arma::file_type::arma_ascii);
    this->alpha.save(filename + "_alpha.txt", arma::file_type::arma_ascii);
}

void FacilityData::FacilityData::read(string filename) {
    arma::vec location_costs_load;
    arma::mat transporation_costs_load;
    arma::vec D_load;
    arma::mat Z_load;
    arma::vec alpha_load;
    location_costs_load.load(filename + "_c.txt", arma::file_type::arma_ascii);
    transporation_costs_load.save(filename + "_c_uv.txt", arma::file_type::arma_ascii);
    D_load.save(filename + "_D.txt", arma::file_type::arma_ascii);
    Z_load.save(filename + "_Z.txt", arma::file_type::arma_ascii);
    alpha_load.save(filename + "_alpha.txt", arma::file_type::arma_ascii);
    this->initialize(location_costs_load, transporation_costs_load, D_load, Z_load, alpha_load);
}



