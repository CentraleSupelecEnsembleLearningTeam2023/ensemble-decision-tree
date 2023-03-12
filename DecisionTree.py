import numpy as np


# Define a node class to represent nodes in the decision tree
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None, is_leaf=False):
        self.feature_index = feature_index  # the index of the feature to split on at this node
        self.threshold = threshold  # the threshold value for the split
        self.left = left  # the left child node
        self.right = right  # the right child node
        self.value = value  # the predicted value for leaf nodes
        self.is_leaf = is_leaf  # whether this node is a leaf node


# Define a base decision tree class
class DecisionTreeBase:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.min_samples_split = min_samples_split  # the minimum number of samples required to split a node
        self.max_depth = max_depth  # the maximum depth of the tree

    # Fit the decision tree to the training data
    def fit(self, X, y):
        self.n_features_ = X.shape[1]  # number of features in the training data
        self.tree = self._grow_tree(X, y)  # grow the decision tree from the training data
        return self
    
    # Make predictions on new data
    def predict(self, X):
        return [self._predict(inputs, self.tree) for inputs in X]
    
    # Recursively traverse the decision tree to make a prediction
    def _predict(self, inputs, node):
        if node.is_leaf:
            return node.value
        if node.feature_index is not None and node.threshold is not None:
            if inputs[node.feature_index] < node.threshold:
                if node.left is not None:
                    return self._predict(inputs, node.left)
                else:
                    return node.value
            else:
                if node.right is not None:
                    return self._predict(inputs, node.right)
                else:
                    return node.value
        else:
            return node.value 