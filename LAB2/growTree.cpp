Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
	
	/* Implement the following:
		--- define stopping criteria
    	--- Loop through candidate features and potential split thresholds.
		--- greedily select the best split according to information gain
		---grow the children that result from the split
	*/


	double best_gain = -1.0; // set the best gain to -1

	int split_idx = -1; // split index
	double split_thresh = 0.0; // split threshold

	//define the stopping criterias for tree growth
	if (depth == 0 || y.size() < min_samples_split) {
		//create a leaf node and assign it most common label
		double predicted_label = DecisionTreeClassification::mostCommonlLabel(y);
		Node* node = new Node(split_idx, split_thresh, NULL, NULL, predicted_label);
		return node;
	}

	//loop through potential features and split thresholds
	//X[0] access the first data point in the set to determine the amount of features this is done to find the size of the features in each data point
	for (int feature = 0; feature < X[0].size(); feature++) {
		for (int i = 0; i < X.size(); i++) {
			double threshold = X[i][feature];
			std::vector<double> X_column;
			
			//calculate the informationgain for the split based on its threshold and feature
			double ig = DecisionTreeClassification::informationGain(y, X[feature], threshold);

			//update the values
			if (ig > best_gain) {
				best_gain = ig;
				split_idx = feature;
				split_thresh = threshold;
			}
		}
	}

	//no split has been found that improves the information gain, create leaf node.
	if (best_gain <= 0.0) {
		double predicted_label = DecisionTreeClassification::mostCommonlLabel(y);
		Node* node = new Node();
		return node;
	}

	//split data based on the best feature and threshold
	std::vector<double> y_right, y_left;
	for (int i = 0; i < y.size(); i++) {
		if (split_thresh >= X[i][split_idx]) {
			y_left.push_back(y[i]);
		}
		else {
			y_right.push_back(y[i]);
		}
	}
	// TODO
	
	Node* left; // grow the left tree
	Node* right;  // grow the right tree
	//grow the tree recursively
	left = growTree(X, y_left, depth - 1);
	right = growTree(X, y_right, depth - 1);
	return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
}
