// Predict method to predict class labels for test data
std::vector<double> LogisticRegression::predict(const std::vector<std::vector<double>>& X_test) {
    std::vector<double> predictions;
        //loop through all vectors in X_test
        for (const std::vector<double>& sample : X_test) {
            std::vector<double> class_probabilities;
            //Loop through all classes in weights vector
            for (int class_label = 1; class_label <= weights.size(); ++class_label) {
                //Calculate the weighted sum of features
                double weighted_sum = 0.0;
                for (double j = 0; j < sample.size(); ++j) {
                    weighted_sum += weights[class_label - 1][j] * sample[j];
                }

                //Calculate the sigmoid of the weighted sum and store it
                double sigmoid_output = sigmoid(weighted_sum);
                class_probabilities.push_back(sigmoid_output);
            }

            /*Find the class with the highest probability and assign it as the prediction
            
            If the if case is not fufilled that mean that the predicted class is equal to 1
            thats why class_label is set to two rather than 1 */
            int predicted_class = 1;
            double max_probability = class_probabilities[0];
            for (int class_label = 2; class_label <= weights.size(); ++class_label) {
                if (class_probabilities[class_label - 1] > max_probability) {
                    predicted_class = class_label;
                    max_probability = class_probabilities[class_label - 1];
                }
            }

            //Store the predicted classes (1, 2, 3) in the predictions vector
            predictions.push_back(static_cast<double>(predicted_class));
        }

        return predictions;
}
