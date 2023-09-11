/// euclideanDistance function: Calculates the Euclidean distance between two vectors.///
double SimilarityFunctions::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;

	for (int i = 0; i < a.size(); i++) {
		//loop each element of vector a and b
		// calculate the difference between corresponding elements of a and b
		// then store it in dist
		dist += pow(a[i] - b[i], 2);
	}
	//square root of dist to obtain the euclidean distance, to obtain the distance
	dist = sqrt(dist);
	// Compute the Euclidean Distance
	// TODO
	
	return dist;
}
