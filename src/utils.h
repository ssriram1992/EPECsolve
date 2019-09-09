#pragma once

#include "epecsolve.h"
#include <armadillo>
#include <fstream>

using namespace std;
using namespace arma;

namespace Utils {
arma::sp_mat resize_patch(const arma::sp_mat &Mat, const unsigned int nR,
                          const unsigned int nC);

arma::mat resize_patch(const arma::mat &Mat, const unsigned int nR,
                       const unsigned int nC);

arma::vec resize_patch(const arma::vec &Mat, const unsigned int nR);

// Saving and retrieving an arma::vec
void appendSave(const vec &matrix, const string out, const string header = "",
                bool erase = false);

long int appendRead(vec &matrix, const string in, long int pos,
                    const string header = "");

// Saving and retrieving an arma::sp_mat
void appendSave(const sp_mat &matrix, const string out,
                const string header = "", bool erase = false);

long int appendRead(sp_mat &matrix, const string in, long int pos,
                    const string header = "");

// Saving and retrieving an std::vector<double>
void appendSave(const vector<double> v, const string out,
                const string header = "", bool erase = false);

long int appendRead(vector<double> &v, const string in, long int pos,
                    const string header = "");

// Saving string
void appendSave(const string v, const string out, bool erase = false);

long int appendRead(string &v, const string in, long int pos);

// Saving A long int
void appendSave(const long int v, const string out, const string header = "",
                bool erase = false);

long int appendRead(long int &v, const string in, long int pos,
                    const string header = "");

// Saving A unsigned int
void appendSave(const unsigned int v, const string out,
                const string header = "", bool erase = false);

long int appendRead(unsigned int &v, const string in, long int pos,
                    const string header = "");

} // namespace Utils
