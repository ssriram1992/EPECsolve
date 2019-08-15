#include"utils.h"
#include<armadillo>
#include<fstream>

using namespace std;
using namespace arma;
// Armadillo patch for inbuild resize
// unsigned uword in row and column indexes create some problems with empty matrix
// which is the case with empty constraints matrices
// For arma::sp_mat
arma::sp_mat Utils::resize_patch(const arma::sp_mat &Mat, const unsigned int nR, const unsigned int nC) {
    arma::sp_mat MMat(nR, nC);
    MMat.zeros();
    bool flag = Mat.n_rows == 7 && Mat.n_cols == 9 && 56 < 5;
    if (nR >= Mat.n_rows && nC >= Mat.n_cols) {
        if (flag) Mat.print_dense("Input");
        if (flag) cout << Mat.n_rows << " " << Mat.n_cols << " ";
        if (Mat.n_rows >= 1 && Mat.n_cols >= 1)
            MMat.submat(0, 0, Mat.n_rows - 1, Mat.n_cols - 1) = Mat;
        if (flag) MMat.print("Output");
    } else {
        if (nR <= Mat.n_rows && nC <= Mat.n_cols)
            MMat = Mat.submat(0, 0, nR, nC);
        else
            throw string(
                    "Error in resize() - the patch for arma::resize. Either both dimension should be larger or both should be smaller!");
    }
    return MMat;
}

// For arma::mat
arma::mat Utils::resize_patch(const arma::mat &Mat, const unsigned int nR, const unsigned int nC) {
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
                    "Error in resize() - the patch for arma::resize. Either both dimension should be larger or both should be smaller!");
    }
    return MMat;
}

// For arma::vec
arma::vec Utils::resize_patch(const arma::vec &Mat, const unsigned int nR) {
    arma::vec MMat(nR);
    MMat.zeros();
    if (nR > Mat.n_rows)
        MMat.subvec(0, Mat.n_rows - 1) = Mat;
    else
        MMat = Mat.subvec(0, nR);
    return MMat;
}


void Utils::appendSave(
		const string in,					///< The input file to be saved
		const string out, 				///< File name of the output file 
		const string header, 			///< A header that might be used to check data correctness
		bool erase					///< Should the matrix be appended to the current file or overwritten
		)
/**
 * Utility to append an arma::sp_mat to a data file.
 */
{
	// Using C++ file operations to copy the data into the target given by @out 

	long size; 			// Number of bytes of data to be written.
	char *buffer; 		// The actual data to be written;

	ifstream infile(in, ios::in);
	ofstream outfile(out, erase?ios::out:ios::app);

	infile.seekg(0, infile.end); 	// Move to the end of the file
	size = infile.tellg(); 			// Current position now, is the number of positions in the file!

	buffer = new char[size]; 		// Now initialize buffer for the correct required size!

	infile.read(buffer, size); 	// Read from the infile 
	infile.close();				// And close it!

	outfile<<header<<"\n"<<size<<"\n"; 	// Write header information
	outfile.write(buffer, size);		// Write the required information of sp_mat
	outfile.close();					// and close it

	delete buffer;
}

void Utils::appendSave(
		const sp_mat &matrix,		///< The arma::sp_mat to be saved
		const string out, 				///< File name of the output file 
		const string header, 			///< A header that might be used to check data correctness
		bool erase					///< Should the matrix be appended to the current file or overwritten
		)
/**
 * Utility to append an arma::sp_mat to a data file.
 */
{
	// Save the matrx to temporary file
	matrix.save("dat/_zz.csv", coord_ascii);
	// Now call the previously defined function.
	Utils::appendSave(string("dat/_zz.csv"), out, header, erase);

}

long int Utils::appendRead(
		const string out, 			///< Read and store the solution in this matrix.
		const string in, 				///< File to read from (could be file with many data is appended one below another)
		long int pos,			///< Position in the long file where reading should start
		const string header 		///< Any header to check data sanctity
		)
/**
 * Utility to read  from a long file and write a small file with an individual data
 * @returns The end position from which the next data object can be read.
 */
{
	long int size; 		// How many bytes to read?
	char * buffer;		// Read to where?
	
	ifstream infile(in, ios::in);
	infile.seekg(pos);

	string header_checkwith;
	infile>>header_checkwith;

	if(header!="" && header != header_checkwith)
		throw string("Error in Utils::appendRead<sp_mat>. Wrong header. Expected: "+header+" Found: "+header_checkwith);

	infile>>size; 	// Get the data size in bytes
	buffer = new char[size];
	infile.read(buffer, size);
	pos = infile.tellg();
	infile.close();

	// Write the data to a file
	ofstream outfile(out, ios::out);
	outfile.seekp(0);
	outfile.write(buffer+1, size - 1); // First character weirdly is a new line character! If problem persists, disable this
	outfile.close();

	return pos;
}


long int Utils::appendRead(
		sp_mat &matrix, 		///< Read and store the solution in this matrix.
		const string in, 				///< File to read from (could be file very many data is appended one below another)
		long int pos,			///< Position in the long file where reading should start
		const string header 		///< Any header to check data sanctity
		)
/**
 * Utility to read an arma::sp_mat from a long file.
 * @returns The end position from which the next data object can be read.
 */
{
	Utils::appendRead(string("dat/_zz.csv"), in, pos, header);

	// Now use armadillo inbuilt function to read from dat/_zz.csv!
	matrix.load("dat/_zz.csv");

	return pos;
}

void appendSave(const vector<double> v, const string out, const string header, bool erase)
{ 
	ofstream outfile(out, erase?ios::out:ios::app);
	outfile<<header<<"\n"<<v.size()<<"\n";
	for(const double x:v)
		outfile<<x<<"\n";
	outfile.close();
}

long int appendRead(vector<double> &v, const string in, long int pos, const string header)
{
	unsigned long int size;
	ifstream infile(in, ios::in);
	infile.seekg(pos);

	string header_checkwith;
	infile>>header_checkwith;

	if(header!="" && header != header_checkwith)
		throw string("Error in Utils::appendRead<sp_mat>. Wrong header. Expected: "+header+" Found: "+header_checkwith);
	
	infile>>size;

	v.resize(size);
	for(unsigned int i=0; i<size; ++i)
		infile>>v[i];
	pos = infile.tellg();
	infile.close();
	return pos;
}
