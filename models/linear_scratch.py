import numpy as np
class LinearRegressionScratch:
    def __init__(self, lr=0.001, epochs=5000):
        self.lr = lr
        self.epochs = epochs
        self.theta = None
        self.loss_history = []
    def fit(self, X, y):
        m, n = X.shape
        # Add bias column
        X_b = np.c_[np.ones((m, 1)), X]
        # Initialize weights randomly
        self.theta = np.random.randn(n + 1, 1)
        for i in range(self.epochs):
            gradients = (2/m) * X_b.T.dot(X_b.dot(self.theta) - y)
            self.theta -= self.lr * gradients
            # Track loss (MSE)
            loss = np.mean((X_b.dot(self.theta) - y) ** 2)
            self.loss_history.append(loss)
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)
