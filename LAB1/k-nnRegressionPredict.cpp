/// predict function: Calculates the predicted values for a given set of test data points using KNN Regression. ///
std::vector<double> KNNRegression::predict(const std::vector<std::vector<double>>& X_test) const {
	std::vector<double> y_pred; // Store predicted values for all test data points
	y_pred.reserve(X_test.size()); // Reserve memory for y_pred to avoid frequent reallocation

	// Check if training data is empty
	if (X_train_.empty() || y_train_.empty()) {
		throw std::runtime_error("Error: Empty training data.");
	}

	/* Implement the following:
		--- Loop through each test data point
		--- Calculate Euclidean distance between test data point and each training data point
		--- Loop through the labels and their counts to find the most frequent label
		--- Store sum of y_train values for k-nearest neighbors
		--- Calculate average of y_train values for k-nearest neighbors
	*/
    
    //loop through each data point, 
    for (const std::vector<double>& data_point : X_test) {
        //create map of distance to label
        std::map<double, double> distance_label_map;
        //calculate distances between current data point and training datapoints

        for (int i = 0; i < X_train_.size(); i++) {
            //access teh current trainting data point from x_train store
            const std::vector<double>& train_point = X_train_[i];
            //calculates the euclidean distance from the current data point and train pont
            double distance = SimilarityFunctions::euclideanDistance(data_point, train_point);
            //store the label associated with current training data point from y_train
            double label = y_train_[i];

            //store the distance-label pairs
            distance_label_map[distance] = label;
        }

        //the value k can be changed 
        //store the sum of label
        int k = 5; 
        double label_sum = 0.0;

        //loop through the distance label map to find the k-nn and calculate their label sum
        for (auto iterator = distance_label_map.begin(); iterator != distance_label_map.end() && k > 0; iterator++, k--) {
            //access the label associated with current neightbor from the distance label map
            double label = iterator->second;
            //add the label of the current neighbour into label sum
            label_sum += label;
        }
        //calculate the average label among the k-nn
        double average_label = label_sum / (3 - k); 
        //store in calculated average label as the prediction for curretn data point
        //y-pred add into average label, which stores predicted vlaues for data points
        y_pred.push_back(average_label);
    }
	//TODO
	return y_pred; 
}
