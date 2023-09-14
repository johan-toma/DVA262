double SimilarityFunctions::manhattanDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;

  
	for (int i = 0; i < a.size(); i++) {
    //loop through each element of a and b
    //subtract each with each othewr and check if result is negative
    //if negative add negative, then add result to dist
		if ((a[i] - b[i]) < 0) dist += -(a[i] - b[i]);
		else dist += a[i] - b[i];
	}

	return dist;
}


	/*double tmp;
	
	for (int i = 0; i < a.size(); i++) {
		tmp = a[i] - b[i];
		if (tmp < 0) tmp = -tmp;
		dist += tmp;
	}*/
