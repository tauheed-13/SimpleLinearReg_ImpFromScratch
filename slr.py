import numpy as np

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.n = 0
    
    def fit(self, X, y):
        self.weights = 0
        self.bias = 0
        self.n = len(X)

        for _ in range(self.iterations):
            y_pred = self.weights*X + self.bias

            dw = (1/self.n)*np.sum(X*(y_pred - y))
            db = (1/self.n)*np.sum(y_pred - y)

            self.weights = self.weights - self.learning_rate*dw
            self.bias = self.bias - self.learning_rate*db
    
    def predict(self, x):
        return self.weights*x + self.bias
    
if __name__ == "__main__":
    X = np.array([1,2,3,4,5])
    y = np.array([5,7,9,11,12])

    model = SimpleLinearRegression()
    model.fit(X, y)
    
    predictions = model.predict(np.array([6, 7]))
    print(f"Predictions for 6 and 7: {predictions}")
    print(f"Calculated Weight: {model.weights:.2f}, Bias: {model.bias:.2f}")
