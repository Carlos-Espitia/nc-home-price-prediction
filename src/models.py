"""From-scratch models (adapted from my ITCS 3156 homework) and evaluation helpers.

All models predict log(price). `evaluate` converts back to dollars for reporting.
"""

import numpy as np


# ---------------------------------------------------------------- helpers
def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones (from the Derivatives & OLS HW)."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2))


def evaluate(log_y: np.ndarray, log_y_hat: np.ndarray) -> dict:
    """Metrics on the log scale (what the models optimize) and in dollars (what people care about).

    exp(predicted log price) estimates the *median* price for a home like this, which is the
    natural "typical price" reading for a skewed target.
    """
    y, y_hat = np.exp(log_y), np.exp(log_y_hat)
    pct_err = np.abs(y_hat - y) / y
    return {
        "RMSE (log)": rmse(log_y, log_y_hat),
        "R² (log)": r2(log_y, log_y_hat),
        "MAE ($)": float(np.mean(np.abs(y_hat - y))),
        "RMSE ($)": rmse(y, y_hat),
        "R² ($)": r2(y, y_hat),
        "Median % error": float(np.median(pct_err) * 100),
        "Within 10%": float(np.mean(pct_err <= 0.10) * 100),
        "Within 20%": float(np.mean(pct_err <= 0.20) * 100),
    }


# ---------------------------------------------------------------- linear models
class OrdinaryLeastSquares:
    """Closed-form linear regression: w = (XᵀX)⁺ Xᵀy (from the Derivatives & OLS HW)."""

    def __init__(self):
        self.w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinaryLeastSquares":
        Xb = add_bias(X)
        self.w = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return add_bias(X) @ self.w


class RidgeOrdinaryLeastSquares:
    """Closed-form ridge regression: w = (XᵀX + λI)⁺ Xᵀy (from the Curve Fitting & Regularization HW).

    The bias term is not penalized (its diagonal entry in I is 0), so lambda only shrinks
    the feature weights, not the overall price level.
    """

    def __init__(self, lamb: float):
        self.lamb = lamb
        self.w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeOrdinaryLeastSquares":
        Xb = add_bias(X)
        I = np.eye(Xb.shape[1])
        I[0, 0] = 0
        self.w = np.linalg.pinv(Xb.T @ Xb + self.lamb * I) @ Xb.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return add_bias(X) @ self.w


def get_batches(n: int, batch_size: int, rng: np.random.RandomState) -> list[np.ndarray]:
    """Shuffled index batches (from the GD & LMS HW)."""
    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


class LeastMeanSquares:
    """Linear regression trained with mini-batch gradient descent (from the GD & LMS HW).

    Loss: MSE = mean((Xw - y)²). Gradient: (2/m) Xᵀ(Xw - y).
    The update w ← w - α·gradient is repeated for every mini-batch, for `epochs` passes.
    """

    def __init__(self, alpha: float, batch_size: int = 64, epochs: int = 30, seed: int = 0):
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.w: np.ndarray | None = None
        self.trn_error: list[float] = []
        self.val_error: list[float] = []

    def fit(self, X, y, X_val=None, y_val=None) -> "LeastMeanSquares":
        rng = np.random.RandomState(self.seed)
        Xb = add_bias(X)
        self.w = rng.rand(Xb.shape[1]) * 0.01
        self.trn_error, self.val_error = [], []
        for _ in range(self.epochs):
            for b in get_batches(len(Xb), self.batch_size, rng):
                Xbat, ybat = Xb[b], y[b]
                grad = 2 / len(b) * Xbat.T @ (Xbat @ self.w - ybat)
                self.w -= self.alpha * grad
            self.trn_error.append(rmse(y, self.predict(X)))
            if X_val is not None:
                self.val_error.append(rmse(y_val, self.predict(X_val)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return add_bias(X) @ self.w


# ---------------------------------------------------------------- KNN
class KNearestNeighborsRegressor:
    """KNN for regression (adapted from the Linear Algebra & KNN HW).

    Same idea as the classifier (find the k closest training homes by Euclidean distance)
    but predicts the AVERAGE of their targets instead of a majority vote.
    Distances are computed in chunks with ||a-b||² = ||a||² + ||b||² - 2a·b so 6,000 x 28,000
    comparisons fit in memory.
    """

    def __init__(self, k: int, chunk: int = 500):
        self.k = k
        self.chunk = chunk
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighborsRegressor":
        self.X, self.y = X, y
        self._sq = np.sum(X ** 2, axis=1)
        return self

    def neighbors(self, X: np.ndarray, k: int, skip_self: bool = False) -> np.ndarray:
        """Indices of the k nearest training rows for each row of X, sorted nearest first.
        skip_self=True drops the closest match (use when X is the training set itself)."""
        kk = k + int(skip_self)
        out = np.empty((len(X), kk), dtype=int)
        for s in range(0, len(X), self.chunk):
            A = X[s:s + self.chunk]
            d2 = np.sum(A ** 2, axis=1)[:, None] + self._sq[None, :] - 2 * A @ self.X.T
            part = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
            order = np.argsort(np.take_along_axis(d2, part, axis=1), axis=1)
            out[s:s + self.chunk] = np.take_along_axis(part, order, axis=1)
        return out[:, 1:] if skip_self else out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.y[self.neighbors(X, self.k)].mean(axis=1)

    def predict_many_k(self, X: np.ndarray, ks: list[int], skip_self: bool = False) -> dict:
        """Predictions for several k values from one neighbor search (for tuning)."""
        nb = self.neighbors(X, max(ks), skip_self=skip_self)
        targets = self.y[nb]
        return {k: targets[:, :k].mean(axis=1) for k in ks}
