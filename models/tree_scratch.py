import numpy as np

class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=5, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def mse(self, y):
        return np.var(y) * len(y)

    def best_split(self, X, y):
        best_feature, best_threshold, best_score = None, None, float('inf')
        n_features = X.shape[1]
        for feat in range(n_features):
            for thr in np.unique(X[:, feat]):
                left = X[:, feat] <= thr
                right = ~left
                if left.sum() < 2 or right.sum() < 2:
                    continue
                score = self.mse(y[left]) + self.mse(y[right])
                if score < best_score:
                    best_score = score
                    best_feature = feat
                    best_threshold = thr
        return best_feature, best_threshold

    def build(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return np.mean(y)

        feat, thr = self.best_split(X, y)
        if feat is None:
            return np.mean(y)

        left = X[:, feat] <= thr
        right = ~left

        return {
            "feature": feat,
            "threshold": thr,
            "left": self.build(X[left], y[left], depth+1),
            "right": self.build(X[right], y[right], depth+1)
        }

    def fit(self, X, y):
        self.tree = self.build(X, y)

    def _predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(row, self.tree) for row in X]).reshape(-1,1)
