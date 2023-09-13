// predict function: Predicts the labels of a given set of test data points using the K-Nearest Neighbors algorithm.//
std::vector<double> KNNClassifier::predict(const std::vector<std::vector<double>>& X_test) const {
	std::vector<double> y_pred; // Store predicted labels for all test data points
	y_pred.reserve(X_test.size()); // Reserve memory for y_pred to avoid frequent reallocation

	// Check if training data is empty
	if (X_train_.empty() || y_train_.empty()) {
		throw std::runtime_error("Error: Empty training data.");
	}


	/* Implement the following:
		--- Loop through each test data point
		--- Calculate Euclidean distance between test data point and each training data point
		--- Loop through the labels and their counts to find the most frequent label
		--- Check if predicted label is valid
	*/

    //loop through each data point, each element of x-test
    for (const std::vector<double>& data_point: X_test) {
        //create map to store the distance label-pairs for the data points
        std::map<double, double> distance_label_map;
        for (int i = 0; i < X_train_.size(); ++i) {
            //get current data point
            const std::vector<double>& train_point = X_train_[i];
            //calculate the euclideandistance between current data point and current training point
            double distance = SimilarityFunctions::euclideanDistance(data_point, train_point); 
            //get the label associated with current training data point
            double label = y_train_[i];
            //then store the calculated distance-label pair in the distance-label map
            distance_label_map[distance] = label;
            
        }
        //k can be adjusted as needed
        int k = 5;
        std::map<double, int> label_count;

        //it iterates through the distance-label-map which contains pairs of distances and labels representing the distance between
        //current datapoint and training data points
        for (auto i = distance_label_map.begin(); i != distance_label_map.end() && k > 0; i++, k--) {
            //acesss to label associated to current distance in map
            double label = i->second;
            //foreach iteration it takes the label assosiated with current distance and increments the label_count
            //basically it will count the amount of times each label appears among k-nn of the data points
            label_count[label]++;
        }
        //find label with the highest count init max count and predict label with false labels -1
        double predicted_label = -1; 
        int max_count = -1;

        //find the most common label among the k-nn
        for (const auto& pair : label_count) {
            //iterate through each label count pair in the label count map
            if (pair.second > max_count) {
                //check if second element is larger then maxcount
                //if greater then update the max-count with the label pair second element
                max_count = pair.second;
                //update the predicted label with the label pair first element
                predicted_label = pair.first;
            }
        }
        //check if predicted label is correct/valid
        if (predicted_label == -1) {
            throw std::runtime_error("Error: Failed to predict a label.");
        }

        y_pred.push_back(predicted_label);
    }

	//TODO
	return y_pred; // Return vector of predicted labels for all test data points
}
