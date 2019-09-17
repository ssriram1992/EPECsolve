#pragma once

#include "epecsolve.h"
#include <armadillo>
#include <fstream>

using namespace arma;

namespace Utils {
arma::sp_mat resize_patch(const arma::sp_mat &Mat, const unsigned int nR,
                          const unsigned int nC);

arma::mat resize_patch(const arma::mat &Mat, const unsigned int nR,
                       const unsigned int nC);

arma::vec resize_patch(const arma::vec &Mat, const unsigned int nR);

// Saving and retrieving an arma::vec
void appendSave(const vec &matrix, const std::string out, const std::string header = "",
                bool erase = false);

long int appendRead(vec &matrix, const std::string in, long int pos,
                    const std::string header = "");

// Saving and retrieving an arma::sp_mat
void appendSave(const sp_mat &matrix, const std::string out,
                const std::string header = "", bool erase = false);

long int appendRead(sp_mat &matrix, const std::string in, long int pos,
                    const std::string header = "");

// Saving and retrieving an std::vector<double>
void appendSave(const std::vector<double> v, const std::string out,
                const std::string header = "", bool erase = false);

long int appendRead(std::vector<double> &v, const std::string in, long int pos,
                    const std::string header = "");

// Saving std::string
void appendSave(const std::string v, const std::string out, bool erase = false);

long int appendRead(std::string &v, const std::string in, long int pos);

// Saving A long int
void appendSave(const long int v, const std::string out, const std::string header = "",
                bool erase = false);

long int appendRead(long int &v, const std::string in, long int pos,
                    const std::string header = "");

// Saving A unsigned int
void appendSave(const unsigned int v, const std::string out,
                const std::string header = "", bool erase = false);

long int appendRead(unsigned int &v, const std::string in, long int pos,
                    const std::string header = "");

} // namespace Utils
