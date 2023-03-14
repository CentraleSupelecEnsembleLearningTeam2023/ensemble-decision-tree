# Implementing a Decision Tree from Scratch
# Decision Tree

A Python package for building decision trees for classification and regression tasks.


## Download

Download the package through git


## Installation

You can install the package using `pip`:

```python
cd directory_of_package
pip install decisiontree-0.0.1-py3-none-any.whl

```


## Usage


### General

Here's an example of how to use the package to build a decision tree:

```python
from decisiontree import DecisionTreeClassification, DecisionTreeRegression

# Load data
X = [[0, 0], [1, 1]]
y = [0, 1]

# Create decision tree classifier
clf = DecisionTreeClassification()
clf.fit(X, y)

# Make predictions
y_pred = clf.predict([[2, 2], [-1, -1]])

```


An example notebook is available inside the example folder.


### Hyperparameters

Both the decision tree for classification and regression can be tuned.
Two parameters are available:
- min_samples_split: minimum number of samples for the leaves
- max_depth: maximum depth for the tree

By default, these are set at 2 for both parameters.

## Contributing

If you find any bugs or would like to suggest new features, please create a GitHub issue or submit a pull request.

## License

This package is licensed under the MIT License. See the LICENSE file for more information