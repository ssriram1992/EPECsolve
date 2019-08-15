#ifndef UTILS_H
#define UTILS_H
#include"epecsolve.h"
#include<armadillo>
#include<fstream>

using namespace std;
using namespace arma;

namespace Utils{
    arma::sp_mat resize_patch(const arma::sp_mat &Mat, const unsigned int nR, const unsigned int nC); 
    arma::mat resize_patch(const arma::mat &Mat, const unsigned int nR, const unsigned int nC); 
    arma::vec resize_patch(const arma::vec &Mat, const unsigned int nR);

	void appendSave(const string in, const string out, const string header="", bool erase=false);
	long int appendRead(const string out, const string in, long int pos, const string header="");

	void appendSave(const vec &matrix, const string out, const string header="", bool erase=false);
	long int appendRead(vec &matrix, const string in, long int pos, const string header="");

	void appendSave(const sp_mat &matrix, const string out, const string header="", bool erase=false);
	long int appendRead(sp_mat &matrix, const string in, long int pos, const string header="");

	void appendSave(const vector<double> v, const string out, const string header = "", bool erase=false);
	long int appendRead(vector<double> &v, const string in, long int pos, const string header="");
};

#endif
