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

# Define a decision tree class for classification     
class DecisionTreeClassification(DecisionTreeBase):
    # Parameters:
    # max_depth(int, default=2): The maximum depth of the tree.
    # min_samples_split(int, default=2): The minimum number of samples required to split an internal node
    
    def __init__(self, min_samples_split=2, max_depth=2):
        super().__init__(min_samples_split=min_samples_split, max_depth=max_depth)
    
    # Fit the decision tree to the training data
    def fit(self, X, y):
        # Set the number of features and classes
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(set(y))
        # Build the tree recursively
        self.tree = self._grow_tree(X, y)
        return self
    
    def _gini(self, y):
        # Calculate the Gini impurity of a set of labels
        classes = np.unique(y)
        n_samples = len(y)
        gini = 1.0
        for c in classes:
            p = len(y[y == c]) / n_samples
            gini -= p ** 2
        return gini
    
    def _best_split(self, X, y):
        # Find the best feature and threshold to split the data
        m = y.size
        if m <= self.min_samples_split:
            return None, None
        # Calculate the Gini impurity of the parent node
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        # Iterate over the features
        for idx in range(self.n_features_):
            # Sort the data points by the feature value
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            # Initialize the number of samples in the left and right nodes
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            # Iterate over the data points
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                # Calculate the Gini impurity of the left and right nodes
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                # Calculate the weighted average of the Gini impurity
                gini = (i * gini_left + (m - i) * gini_right) / m
                # Skip duplicates
                if thresholds[i] == thresholds[i - 1]:
                    continue
                # Update the best split if necessary
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr
    
    def _grow_tree(self, X, y, depth=0):
        # Calculate number of samples per class
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        # Predict the class with maximum number of samples
        predicted_class = np.argmax(num_samples_per_class)
        # Create a node with the predicted class as its value
        node = Node(value=predicted_class)

        # Check if tree depth is less than maximum depth
        if depth < self.max_depth:
            # Find the best feature and threshold for splitting the data
            idx, thr = self._best_split(X, y)
            if idx is not None:
                # Split the data based on the feature and threshold
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]

                # Check if there are samples in both the left and right child nodes
                if len(X_left) > 0 and len(X_right) > 0:
                    # Recursively grow the left and right child nodes
                    child_left = self._grow_tree(X_left, y_left, depth+1)
                    child_right = self._grow_tree(X_right, y_right, depth+1)
                    # Set the feature index and threshold for the current node
                    node.feature_index = idx
                    node.threshold = thr
                    # Set the left and right child nodes for the current node
                    node.left = child_left
                    node.right = child_right
                    return node

        # If the tree depth is greater than or equal to the maximum depth, or if the best split could not be found,
        # set the current node as a leaf node
        node.is_leaf = True
        return node

# Define a decision tree class for regression
class DecisionTreeRegression(DecisionTreeBase):
    # Parameters:
    # max_depth(int, default=2): The maximum depth of the tree.
    # min_samples_split(int, default=2): The minimum number of samples required to split an internal node
    def __init__(self, min_samples_split=2, max_depth=2):
        # Initialize DecisionTreeBase with min_samples_split and max_depth
        super().__init__(min_samples_split=min_samples_split, max_depth=max_depth)

    def _mse(self, y):
        # Calculate the mean squared error of a set of target values
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _best_split(self, X, y):
        # Find the best split for the given data and target values
        m = y.size
        if m <= self.min_samples_split:
            # If the number of samples is less than or equal to min_samples_split, return None
            return None, None
        best_mse = np.inf
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            # Sort the feature values and target values by the feature values
            thresholds, values = zip(*sorted(zip(X[:, idx], y)))
            for i in range(1, m):
                if thresholds[i] == thresholds[i - 1]:
                    continue
                # Calculate the mean squared error for the left and right subsets of the data
                mse_left = self._mse(values[:i])
                mse_right = self._mse(values[i:])
                # Calculate the weighted mean squared error for the split
                mse = (i * mse_left + (m - i) * mse_right) / m
                if mse < best_mse:
                    # If the weighted mean squared error is better than the previous best, update the best values
                    best_mse = mse
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        # Return the best feature index and threshold for the split
        return best_idx, best_thr
        
    def _grow_tree(self, X, y, depth=0):
        # Recursively grow the decision tree
        node = Node(value=np.mean(y))
        if depth < self.max_depth:
            # Find the best split for the current data and target values
            idx, thr = self._best_split(X, y)
            if idx is not None:
                # Split the data and target values based on the best split
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                if len(X_left) > 0 and len(X_right) > 0:
                    # Recursively grow the left and right subtrees
                    child_left = self._grow_tree(X_left, y_left, depth+1)
                    child_right = self._grow_tree(X_right, y_right, depth+1)
                    # Update the node with the best feature index, threshold, and left and right subtrees
                    node.feature_index = idx
                    node.threshold = thr
                    node.left = child_left
                    node.right = child_right
                    return node
        # If the maximum depth has been reached or the best split cannot be found, make the node a leaf node
        node.is_leaf = True
        return node