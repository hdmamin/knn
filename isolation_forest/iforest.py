# Algorithm: https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
import math
import numpy as np
import pandas as pd
import random


class IsolationTreeEnsemble:
    """An Isolation Forest for anomaly detection."""

    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.max_depth = np.ceil(np.log2(self.sample_size))
        self.trees = [IsolationTree(self.max_depth) for i in range(n_trees)]
        self.n = None

    def fit(self, X, improved=False):
        """Given a 2D matrix of observations, create an ensemble of
        IsolationTree objects and store them in a list: self.trees. Convert
        DataFrames to ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.n = X.shape[0]

        # Sample with replacement, then fit tree.
        for tree in self.trees:
            x_idx = random.sample(range(self.n), self.sample_size)
            tree.fit(X[x_idx], improved)

    def path_length(self, X):
        """Given a 2D matrix of observations, X, compute the average path
        length for each observation in X.  Compute the path length for x_i
        using every tree in self.trees then compute the average for each x_i.
        Return an ndarray of shape (len(X),1).
        """
        c = np.zeros((X.shape[0], self.n_trees))
        for i, x_i in enumerate(X):
            for j, tree in enumerate(self.trees):
                c[i, j] = tree._path_length(tree.root, x_i, 0)
        return np.mean(c, axis=1, keepdims=True)

    def anomaly_score(self, X):
        """Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.

        h is the average path length for each x from all trees in the forest.
        c_norm is a scalar used to normalize the scores.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        h = self.path_length(X)
        c_norm = c(self.sample_size)
        return 2 ** -(h/c_norm)

    def predict_from_anomaly_scores(self, scores, threshold):
        """Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return np.where(scores >= threshold, 1, 0)

    def predict(self, X, threshold: float):
        """A shorthand for calling anomaly_score() and
        predict_from_anomaly_scores().
        """
        scores = self.anomaly_score(X)
        return self.predict_from_anomaly_scores(scores, threshold)

    def __str__(self):
        return ('-'*60 + '\nIsolationTreeEnsemble\n' + '-'*60 + '\n' +
                '\n\n'.join([str(t) for t in self.trees])) + '\n' + '-'*60

    def __repr__(self):
        return str(self)


class IsolationTree:
    """One tree in an isolation forrest for anomaly detection."""

    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.root = None
        self.n_nodes = 0
        self.c_cache = dict()

    def fit(self, X, improved=False, node_depth=0):
        """Recursively split data based on a random feature and random split
        point.
        """
        # Node count reflects count once function call completes.
        self.n_nodes += 1
        nrows, ncols = X.shape
        if nrows <= 1 or node_depth == self.height_limit:
            return LeafNode(nrows)

        # Choose split column and value.
        split_col = np.random.randint(ncols)
        x_col = X[:, split_col]
        x_min, x_max = x_col.min(), x_col.max()
        if improved:
            split_point = self._improved_split(x_col, x_min, x_max)
        else:
            split_point = random.uniform(x_min, x_max)
        x_left = X[x_col < split_point]
        x_right = X[x_col >= split_point]
        self.root = DecisionNode(self.fit(x_left, improved, node_depth+1),
                                 self.fit(x_right, improved, node_depth+1),
                                 split_col, split_point)
        return self.root

    def _improved_split(self, x_col, col_min, col_max, n=2):
        """Find better split if improved functionality is called. Randomly
        choose 3 columns and generate 1 potential split point for each, then
        choose the split that minimizes P*(1-P), where P is the percent
        of x values on the left and 1-P is the percent on the right. This will
        minimized when P approaches 0 or 1, meaning when we are closer to
        isolating an x value. We adjust the calculation in the rare case when
        all x values go to the same side, since that does not come closer to
        isolating any points.
        """
        split_points = np.random.uniform(col_min, col_max, size=n)
        scores = []
        for point in split_points:
            p_left = np.mean(x_col < point)
            # Want 1 side w/ few points but not zero points.
            if p_left == 0 or p_left == 1:
                scores.append(1)
            else:
                scores.append(p_left * (1-p_left))
        return split_points[np.argmin(scores)]

    def _path_length(self, node, x_i, length):
        """Find and return the path length to example x_i. Length should be 0
         when first called.
        """
        while node.left is not None:
            length += 1
            if x_i[node.split_col] < node.split_val:
                node = node.left
            else:
                node = node.right
        if node.rows in self.c_cache:
            return length + self.c_cache[node.rows]
        else:
            self.c_cache[node.rows] = tmp = c(node.rows)
            return length + tmp

    def __str__(self):
        return f'IsolationTree{self.root}'

    def __repr__(self):
        return str(self)


class DecisionNode:
    """Internal node in an IsolationTree."""

    def __init__(self, left, right, split_col, split_val):
        """Create a node in an isolation tree.

        split_col : int
            Contains index of column to split on.
        split_val : float
            Value in split_col where split occurs.
        """
        self.left = left
        self.right = right
        self.split_col = split_col
        self.split_val = split_val

    def __str__(self):
        return (f'\n\tDecisionNode(col {self.split_col}: {self.split_val:.3f},'
                f'\n\t\tleft={self.left},\n\t\tright={self.right})')

    def __repr__(self):
        return str(self)


class LeafNode:
    """Leaf node in an IsolationTree."""

    def __init__(self, rows, left=None, right=None):
        self.rows = rows
        self.left = None
        self.right = None

    def __str__(self):
        return f'\n\t\tLeafNode(size: {self.rows})'

    def __repr__(self):
        return str(self)


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    if isinstance(y, pd.Series):
        y = y.values
    y = y.reshape(scores.shape)
    for thresh in np.arange(1, 0, -.01):
        rounded_scores = np.where(scores >= thresh, 1, 0)
        # Max in denominator prevents us from dividing by zero.
        tpr = (np.logical_and(y, rounded_scores).sum() / max(sum(y), 1))
        fpr = (np.sum(rounded_scores - y == 1) / max(sum(1 - y), 1))
        if tpr >= desired_TPR:
            return thresh, np.squeeze(fpr)


def c(size):
    """Compute value to normalize path length when computing anomaly
    scores.
    """
    if size > 2:
        val = 2 * (math.log(size-1) + np.euler_gamma) - 2 * (size-1) / size
    elif size == 2:
        val = 1
    else:
        val = 0
    return val
