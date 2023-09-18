/// informationGain function: Calculates the information gain of a given split threshold for a given feature column.
double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
	// parent loss // You need to caculate entropy using the EntropyFunctions class//
	double parent_entropy = EntropyFunctions::entropy(y);

	/* Implement the following:
	   --- generate split
	   --- compute the weighted avg. of the loss for the children
	   --- information gain is difference in loss before vs. after split
	*/
	double ig = 0.0;
	
	// TODO

	//seperate the data into two child datasets
	std::vector<double> y_leftchild;
	std::vector<double> y_rightchild;

	//split the children based on the split_thresh
	for (int i = 0; i < y.size(); i++) {
		if (X_column[i] <= split_thresh) {
			y_leftchild.push_back(y[i]);
		}
		else {
			y_rightchild.push_back(y[i]);
		}
	}

	//calculate the entropy for left and right children nodes
	double entropy_leftchild = EntropyFunctions::entropy(y_leftchild);
	double entropy_rightchild = EntropyFunctions::entropy(y_rightchild);

	//calculate the weighted average of the children entropies
	//relative weight of the left children dataset compared to parent dataset
	double left_weight = y_leftchild.size() / y.size();
	//relative weight of the right children dataset compared to parent dataset
	double right_weight = y_rightchild.size() / y.size();
	double child_average_entropy = left_weight * entropy_leftchild + entropy_rightchild * right_weight;
	
	//here is calculation for information gain basically its the difference between the parent entropy and the weighted average of the childrens entropies.
	ig = parent_entropy - child_average_entropy;
	return ig;
}
