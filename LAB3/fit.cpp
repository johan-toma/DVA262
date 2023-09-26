// Fit method for training the logistic regression model
void LogisticRegression::fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) {
    int num_features = X_train[0].size();
    int num_classes = std::set<double>(y_train.begin(), y_train.end()).size();

    //Initialize weights top vector with size "num_classes" and sub vectors with size "num_features" with value ?
    weights = std::vector<std::vector<double>>(num_classes, std::vector<double>(num_features, 0.5));
    
	/* Implement the following:
       	--- Initialize weights for each class
    	--- Loop over each class label
    	--- Convert the problem into a binary classification problem
        --- Loop over training epochs
       	--- Add bias term to the training example
    	--- Calculate weighted sum of features
        --- Calculate the sigmoid of the weighted sum
        --- Update weights using gradient descent
    */
    for (double class_label = 1; class_label <= num_classes; ++class_label) {
        //Converting to binary for each class (1,2,3) -> (1,0)
        std::vector<double> binaryLabels;
        for (double label : y_train) {
            //add 1 if label is equal to class_label
            if (label == class_label) {
                binaryLabels.push_back(1);  
            }
            //Otherwise add 0
            else {
                binaryLabels.push_back(0);  
            }
        }
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            for (size_t i = 0; i < X_train.size(); ++i) {
                // Add bias term to the training example (assuming X_train already has it)
                const std::vector<double>& features = X_train[i];

                // Calculate the weighted sum of features
                double weighted_sum = 0.0;
                for (size_t j = 0; j < num_features; ++j) {
                    weighted_sum += weights[class_label - 1][j] * features[j];
                }

                //Calculate the sigmoid using weighted sum
                double sigmoid_output = sigmoid(weighted_sum);

                // Update weights using gradient descent
                for (size_t j = 0; j < num_features; ++j) {
                    double diff = binaryLabels[i] - sigmoid_output;
                    weights[class_label - 1][j] += learning_rate * diff * sigmoid_output * (1.0 - sigmoid_output) * features[j];
                }
            }
        }
    }
    
}
