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