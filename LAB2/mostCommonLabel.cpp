// mostCommonlLabel function: Finds the most common label in a vector of labels.//
double DecisionTreeClassification::mostCommonlLabel(std::vector<double>& y) {	
	double most_common = 0.0;
	
	// TODO
	//assume the first in the vector of labels, the first label is the most common label and store its count
	int common_count = 1;
	most_common = y[0];

	//go through the labels from the second element
	for (int i = 1; i < y.size(); i++) {
		double cur_label = y[i];
		int cur_count = 1;
		//count the amount of current label in the vector
		for (int j = i + 1; j < y.size(); j++) {
			if (y[j] == cur_label) {
				cur_count++;
			}
		}
		//replace the old values of most common and its count, with the new most common and new count.
		if (cur_count > common_count) {
			most_common = cur_label;
			common_count = cur_count;
		}
	}
	return most_common;
}
