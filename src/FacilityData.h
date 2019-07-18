#include <iostream>
#include <memory>
#include <gurobi_c++.h>
#include <armadillo>
#include<stdbool.h>

#ifndef EPECSOLVE_FACILITYDATA_H
#define EPECSOLVE_FACILITYDATA_H

namespace FacilityData {

    class FacilityData {
    private:
        int num_locations{0};
        arma::vec location_costs; //
        arma::mat transporation_costs; // Symmetric + diagonal is zero
        int num_leaders{0};
        arma::vec D; //Demand caps for leaders
        arma::mat Z; // Row is the leader index, column is d index
        arma::vec alpha; // Interaction weights for each leader

        virtual FacilityData &initialize(arma::vec &c, arma::mat &c_uv, arma::vec &D, arma::mat &Z,
                                   arma::vec &alpha);


    public:
        FacilityData(arma::vec &&c, arma::mat &&c_uv, arma::vec &&D, arma::mat &&Z,
                     arma::vec &&alpha); //Constructor (move)

        //Getters
        virtual inline int get_num_locations() const final { return this->num_locations; }

        virtual inline int get_num_leaders() const final { return this->num_leaders; }

        virtual inline arma::vec get_location_costs() const final { return this->location_costs; }

        virtual inline arma::mat get_transportation_costs() const final { return this->transporation_costs; }

        virtual inline arma::vec get_D() const final { return this->D; }

        virtual inline arma::mat get_Z() const final { return this->Z; }

        virtual inline arma::vec get_alpha() const final { return this->alpha; }

        void write(string filename) const;

        virtual void read(string filename);
    };

}
#endif //EPECSOLVE_FACILITYDATA_H
