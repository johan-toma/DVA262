// traverseTree function: Traverses a decision tree given an input vector and a node.//
double DecisionTreeClassification::traverseTree(std::vector<double>& x, Node* node) {

	/* Implement the following:
		--- If the node is a leaf node, return its value
		--- If the feature value of the input vector is less than or equal to the node's threshold, traverse the left subtree
		--- Otherwise, traverse the right subtree
	*/
	// TODO
	// 
	//check if node is leafnode return value
	if (node->isLeafNode()) {
		return node->value;
	}
	//traverse left tree
	if (x[node->feature] <= node->threshold) {
		return traverseTree(x, node->left);
	}
	else {
		//traverse right tree
		return traverseTree(x, node->right);
	}
	
	//return 0.0;
}
