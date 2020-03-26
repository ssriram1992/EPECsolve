#include "utils.h"
#include <armadillo>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <fstream>

using namespace std;
using namespace arma;

arma::sp_mat Utils::resize_patch(const arma::sp_mat &Mat, const unsigned int nR,
                                 const unsigned int nC) {
  /**
 @brief Armadillo patch for resizing sp_mat
 @details Armadillo sp_mat::resize() is not robust as it initializes garbage
 values to new columns. This fixes the problem by creating new columns with
 guaranteed zero values. For arma::sp_mat
 */
  arma::sp_mat MMat(nR, nC);
  MMat.zeros();
  if (nR >= Mat.n_rows && nC >= Mat.n_cols) {
    if (Mat.n_rows >= 1 && Mat.n_cols >= 1)
      MMat.submat(0, 0, Mat.n_rows - 1, Mat.n_cols - 1) = Mat;
  } else {
    if (nR <= Mat.n_rows && nC <= Mat.n_cols)
      MMat = Mat.submat(0, 0, nR, nC);
    else
      throw string(
          "Error in resize() - the patch for arma::resize. Either both "
          "dimension should be larger or both should be smaller!");
  }
  return MMat;
}

// For arma::mat
arma::mat Utils::resize_patch(const arma::mat &Mat, const unsigned int nR,
                              const unsigned int nC) {
  /**
 @brief Armadillo patch for resizing mat
 @details Armadillo mat::resize() is not robust as it initializes garbage
 values to new columns. This fixes the problem by creating new columns with
 guaranteed zero values. For arma::mat
 */
  arma::mat MMat(nR, nC);
  MMat.zeros();
  if (nR >= Mat.n_rows && nC >= Mat.n_cols) {
    if (Mat.n_rows >= 1 && Mat.n_cols >= 1)
      MMat.submat(0, 0, Mat.n_rows - 1, Mat.n_cols - 1) = Mat;
  } else {
    if (nR <= Mat.n_rows && nC <= Mat.n_cols)
      MMat = Mat.submat(0, 0, nR, nC);
    else
      throw string(
          "Error in resize() - the patch for arma::resize. Either both "
          "dimension should be larger or both should be smaller!");
  }
  return MMat;
}

// For arma::vec
arma::vec Utils::resize_patch(const arma::vec &Mat, const unsigned int nR) {
  /**
 @brief Armadillo patch for resizing vec
 @details Armadillo vec::resize() is not robust as it initializes garbage
 values to new columns. This fixes the problem by creating new columns with
 guaranteed zero values. For arma::vec
 */
  arma::vec MMat(nR);
  MMat.zeros();
  if (nR > Mat.n_rows)
    MMat.subvec(0, Mat.n_rows - 1) = Mat;
  else
    MMat = Mat.subvec(0, nR);
  return MMat;
}

void Utils::appendSave(const sp_mat &matrix, ///< The arma::sp_mat to be saved
                       const string out,     ///< File name of the output file
                       const string header,  ///< A header that might be used to
                                             ///< check data correctness
                       bool erase ///< Should the matrix be appended to the
                                  ///< current file or overwritten
                       )
/**
 * Utility to append an arma::sp_mat to a data file.
 */
{
  // Using C++ file operations to copy the data into the target given by @out
  unsigned int nR{0}, nC{0}, nnz{0};

  ofstream outfile(out, erase ? ios::out : ios::app);

  nR = matrix.n_rows;
  nC = matrix.n_cols;
  nnz = matrix.n_nonzero;

  outfile << header << "\n";
  outfile << nR << "\t" << nC << "\t" << nnz << "\n";
  for (auto it = matrix.begin(); it != matrix.end(); ++it)
    outfile << it.row() << "\t" << it.col() << "\t" << (*it)
            << "\n"; // Write the required information of sp_mat
  outfile << "\n";
  outfile.close(); // and close it
}

long int Utils::appendRead(
    sp_mat &matrix,  ///< Read and store the solution in this matrix.
    const string in, ///< File to read from (could be file very many data is
                     ///< appended one below another)
    long int pos,    ///< Position in the long file where reading should start
    const string header ///< Any header to check data sanctity
    )
/**
 * Utility to read an arma::sp_mat from a long file.
 * @returns The end position from which the next data object can be read.
 */
{
  unsigned int nR, nC, nnz;

  ifstream infile(in, ios::in);
  infile.seekg(pos);

  string header_checkwith;
  infile >> header_checkwith;

  if (header != "" && header != header_checkwith)
    throw string(
        "Error in Utils::appendRead<sp_mat>. Wrong header. Expected: " +
        header + " Found: " + header_checkwith);

  infile >> nR >> nC >> nnz;
  if (nR == 0 || nC == 0)
    matrix.set_size(nR, nC);
  else {
    arma::umat locations(2, nnz);
    arma::vec values(nnz);

    unsigned int r, c;
    double val;

    for (unsigned int i = 0; i < nnz; ++i) {
      infile >> r >> c >> val;
      locations(0, i) = r;
      locations(1, i) = c;
      values(i) = val;
    }
    matrix = arma::sp_mat(locations, values, nR, nC);
  }

  pos = infile.tellg();
  infile.close();

  return pos;
}

void appendSave(const vector<double> v, const string out, const string header,
                bool erase) {
  /**
   * Utility to append an std::vector<double> to a data file.
   */
  ofstream outfile(out, erase ? ios::out : ios::app);
  outfile << header << "\n" << v.size() << "\n";
  for (const double x : v)
    outfile << x << "\n";
  outfile.close();
}

long int appendRead(vector<double> &v, const string in, long int pos,
                    const string header) {
  unsigned long int size;
  ifstream infile(in, ios::in);
  infile.seekg(pos);
  /**
   * Utility to read an std::vector<double> from a long file.
   * @returns The end position from which the next data object can be read.
   */

  string header_checkwith;
  infile >> header_checkwith;

  if (header != "" && header != header_checkwith)
    throw string(
        "Error in Utils::appendRead<sp_mat>. Wrong header. Expected: " +
        header + " Found: " + header_checkwith);

  infile >> size;

  v.resize(size);
  for (unsigned int i = 0; i < size; ++i)
    infile >> v[i];
  pos = infile.tellg();
  infile.close();
  return pos;
}

void Utils::appendSave(const vec &matrix,   ///< The arma::vec to be saved
                       const string out,    ///< File name of the output file
                       const string header, ///< A header that might be used to
                                            ///< check data correctness
                       bool erase ///< Should the vec be appended to the
                                  ///< current file or overwritten
) {
  /**
   * Utility to append an arma::vec to a data file.
   */
  // Using C++ file operations to copy the data into the target given by @out
  unsigned int nR{0};

  ofstream outfile(out, erase ? ios::out : ios::app);

  nR = matrix.n_rows;

  outfile << header << "\n";

  outfile << nR << "\n";
  for (auto it = matrix.begin(); it != matrix.end(); ++it)
    outfile << (*it) << "\n"; // Write the required information of sp_mat
  outfile << "\n";
  outfile.close(); // and close it
}

long int Utils::appendRead(
    vec &matrix,     ///< Read and store the solution in this matrix.
    const string in, ///< File to read from (could be file very many data is
                     ///< appended one below another)
    long int pos,    ///< Position in the long file where reading should start
    const string header ///< Any header to check data sanctity
) {
  /**
   * Utility to read an arma::vec from a long file.
   * @returns The end position from which the next data object can be read.
   */
  unsigned int nR;
  string buffers;
  string checkwith;
  ifstream in_file(in, ios::in);
  in_file.seekg(pos);

  in_file >> checkwith;
  if (header != "" && checkwith != header)
    throw string("Error in Utils::appendRead<vec>. Wrong header. Expected: " +
                 header + " Found: " + checkwith);
  in_file >> nR;
  matrix.zeros(nR);
  for (unsigned int i = 0; i < nR; ++i) {
    double val;
    in_file >> val;
    matrix.at(i) = val;
  }

  pos = in_file.tellg();
  in_file.close();

  return pos;
}

void Utils::appendSave(const long int v, const string out, const string header,
                       bool erase)
/**
 * Utility to save a long int to file
 */
{
  ofstream outfile(out, erase ? ios::out : ios::app);
  outfile << header << "\n";
  outfile << v << "\n";
  outfile.close();
}

long int Utils::appendRead(long int &v, const string in, long int pos,
                           const string header) {
  /**
   * Utility to read a long int from a long file.
   * @returns The end position from which the next data object can be read.
   */
  ifstream infile(in, ios::in);
  infile.seekg(pos);

  string header_checkwith;
  infile >> header_checkwith;

  if (header != "" && header != header_checkwith)
    throw string(
        "Error in Utils::appendRead<long int>. Wrong header. Expected: " +
        header + " Found: " + header_checkwith);

  long int val;
  infile >> val;
  v = val;

  pos = infile.tellg();
  infile.close();

  return pos;
}

void Utils::appendSave(const unsigned int v, const string out,
                       const string header, bool erase)
/**
 * Utility to save a long int to file
 */
{
  ofstream outfile(out, erase ? ios::out : ios::app);
  outfile << header << "\n";
  outfile << v << "\n";
  outfile.close();
}

long int Utils::appendRead(unsigned int &v, const string in, long int pos,
                           const string header) {
  ifstream infile(in, ios::in);
  infile.seekg(pos);

  string header_checkwith;
  infile >> header_checkwith;

  if (header != "" && header != header_checkwith)
    throw string(
        "Error in Utils::appendRead<unsigned int>. Wrong header. Expected: " +
        header + " Found: " + header_checkwith);

  unsigned int val;
  infile >> val;
  v = val;

  pos = infile.tellg();
  infile.close();

  return pos;
}

void Utils::appendSave(const string v, const string out, bool erase)
/**
 * Utility to save a long int to file
 */
{
  ofstream outfile(out, erase ? ios::out : ios::app);
  outfile << v << "\n";
  outfile.close();
}

long int Utils::appendRead(string &v, const string in, long int pos) {
  /**
   * Utility to read a std::string from a long file.
   * @returns The end position from which the next data object can be read.
   */
  ifstream infile(in, ios::in);
  infile.seekg(pos);

  string val;
  infile >> val;
  v = val;

  pos = infile.tellg();
  infile.close();

  return pos;
}
unsigned long int Utils::vec_to_num(std::vector<short int> binary) {
  unsigned long int number = 0;
  unsigned int posn = 1;
  while (!binary.empty()) {
    short int bit = (binary.back() + 1) / 2; // The least significant bit
    number += (bit * posn);
    posn *= 2;         // Update place value
    binary.pop_back(); // Remove that bit
  }
  return number;
}

std::vector<short int> Utils::num_to_vec(unsigned long int number,
                                         const unsigned int &nCompl) {
  std::vector<short int> binary{};
  for (unsigned int vv = 0; vv < nCompl; vv++) {
    binary.push_back(number % 2);
    number /= 2;
  }
  std::for_each(binary.begin(), binary.end(),
                [](short int &vv) { vv = (vv == 0 ? -1 : 1); });
  std::reverse(binary.begin(), binary.end());
  return binary;
}