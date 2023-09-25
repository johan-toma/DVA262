// Predict method to predict class labels for test data
std::vector<double> LogisticRegression::predict(const std::vector<std::vector<double>>& X_test) {
    std::vector<double> predictions;
    
    /* Implement the following:
    	--- Loop over each test example
        --- Add bias term to the test example
        --- Calculate scores for each class by computing the weighted sum of features
        --- Predict class label with the highest score
    */
    //random number generator for the weights
    std::random_device rand;
    std::mt19937 generate(rand());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    // TODO
    for (std::vector<double> test : X_test) {
        double score = 0;
        int count = 0;
        std::vector<double> weights(test.size());
        if (count == 0) {
            for (int i = 0; i < test.size(); i++) {
                weights[i] = dist(generate);
            }
            count = 1;
        }

        for (int i = 0; i < test.size(); i++) {
            score = score + weights[i] * test[i];
        }

        double predicted_class_label;
        if (score >= 0.0) {
            predicted_class_label = 1.0;
        }
        else {
            predicted_class_label = 0.0;
        }
    }
    return predictions;
}
