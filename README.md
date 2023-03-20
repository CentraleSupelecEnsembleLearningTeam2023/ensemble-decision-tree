# Implementing a Decision Tree from Scratch
# Decision Tree

A Python package for building decision trees for classification and regression tasks.

## Authors
The project is done by [Karim El Hage](https://github.com/karimelhage), [Ali Najem](https://github.com/najemali), [Annabelle Luo](https://github.com/annabelleluo), [Xiaoyan Hong](https://github.com/EmmaHongW), [Antoine Cloute](https://github.com/AntAI-Git)

<a href="https://github.com/karimelhage/ensemble-decision-tree/graphs/contributors"> 
  <img src="https://contrib.rocks/image?repo=karimelhage/ensemble-decision-tree" />
</a>

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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Fit Model on train set
clf = DecisionTreeClassification()
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

```


An example notebook with different toy datasets is available inside the example folder.


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
