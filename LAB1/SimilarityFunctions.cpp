#include "SimilarityFunctions.h"
#include <cmath>
#include <stdexcept>


										// SimilarityFunctions class implementation //
			

/// hammingDistance function: Calculates the Hamming distance between two vectors.
double SimilarityFunctions::hammingDistance(const std::vector<double>& v1, const std::vector<double>& v2) {
	if (v1.size() != v2.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;
	
	// Compute the Hamming Distance
	//TODO

	return dist;
}


/// jaccardDistance function: Calculates the Jaccard distance between two vectors.
double SimilarityFunctions::jaccardDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double num = 0.0;
	double den = 0.0;
	double dist = 0.0;
	
	// Compute the Jaccard Distance
	// TODO

	return dist;
}


/// cosineDistance function: Calculates the cosine distance between two vectors.///
double SimilarityFunctions::cosineDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dotProduct = 0.0;
	double normA = 0.0;
	double normB = 0.0;
	double cosinedist = 0.0;	
	
	// Compute the cosine Distance
	// TODO

	
	return cosinedist;
}


/// euclideanDistance function: Calculates the Euclidean distance between two vectors.///
double SimilarityFunctions::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;

	//a & b size are the same so use one to loop through 
	for (int i = 0; i < a.size(); i++) {
		//a[i] - b[i] to the power of two
		dist += pow(a[i] - b[i], 2);
	}
	dist = sqrt(dist); //square root of dist to get euclidean distance
	
	
	//dist = sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2) + pow(a[3] - b[3], 2));

	return dist;
}


/// manhattanDistance function: Calculates the Manhattan distance between two vectors.///
double SimilarityFunctions::manhattanDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;
	double tmp;
	
	for (int i = 0; i < a.size(); i++) {
		tmp = a[i] - b[i];
		if (tmp < 0) tmp = -tmp;
		dist += tmp;
	}
	/*for (int i = 0; i < a.size(); i++) {
		if ((a[i] - b[i]) < 0) dist += -(a[i] - b[i]);
		else dist += a[i] - b[i];
	}*/

	/*double x_len, x_wit, y_len, y_wit;

	x_len = a[0] - b[0];
	x_wit = a[1] - b[1];
	y_len = a[2] - b[2];
	y_wit = a[3] - b[3];

	if (x_len < 0) {
		x_len = -x_len;
	}
	if (x_wit < 0) {
		x_wit = -x_wit;
	}
	if (y_len < 0) {
		y_len = -y_len;
	}
	if (y_wit < 0) {
		y_wit = -y_wit;
	}*/

	//dist = x_len + x_wit + y_len + y_wit;

	return dist;
}

/// minkowskiDistance function: Calculates the Minkowski distance between two vectors.///
double SimilarityFunctions::minkowskiDistance(const std::vector<double>& a, const std::vector<double>& b, int p) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;
	
	// Compute the Minkowski Distance
	// TODO
	

	return dist;
}


