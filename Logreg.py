import numpy as np

def logreg_full_batch(X: np.ndarray, Y: np.ndarray, epoch: int, learning_rate=0.1, lambda_reg=0.01) -> tuple[float | np.floating, np.ndarray, float | np.floating]:
    """
    Logistic Regression function using Full-batch Gradient Descent
    """
    m = X.shape[0]
    theta = np.zeros(X.shape[1])
    bias = 0.0

    best_cost = float("inf")
    best_theta = np.copy(theta)
    best_bias = bias
    eps = 1e-15

    while (epoch > 0):
        hofX = sigmoid(theta, X, bias)      # predictions
        error = hofX - Y
        gradient = X.T @ error / m + (lambda_reg / m) * theta   # L2 regularization
        gradient_bias = np.mean(error)

        theta = theta - learning_rate * gradient
        bias = bias - learning_rate * gradient_bias
        
        current_cost = -np.mean(Y * np.log(hofX + eps) + (1 - Y) * np.log(1 - hofX + eps)) + (lambda_reg / (2 * m)) * np.sum(theta ** 2)  # Penalty
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_theta = np.copy(theta)
            best_bias = bias
            print(f"Epoch {epoch}: New best cost = {best_cost:.4f}")
        epoch -= 1
    print(f"Best cost: {best_cost}")
    return best_cost, best_theta, best_bias


def logreg_mini_batch(X: np.ndarray, Y: np.ndarray, epoch=5, batch_size=512, learning_rate=0.1, seed=42, lambda_reg=0.01) -> tuple[float | np.floating, np.ndarray, float | np.floating]:
    """
    Logistic Regression function using mini-batch Gradient Descent
    """
    m = X.shape[0]
    learning_rate = 0.1
    theta = np.zeros(X.shape[1])
    bias = 0.0

    best_cost = float("inf")
    best_theta = np.copy(theta)
    best_bias = bias
    eps = 1e-15

    rng = np.random.default_rng(seed)

    while (epoch > 0):
        perm = rng.permutation(m)

        for start in range(0, m, batch_size):
            idx = perm[start:start + batch_size]
            X_batch = X[idx]
            Y_batch = Y[idx]
            batch_m = X_batch.shape[0]             # in case last batch is smaller in size

            hofX = sigmoid(theta, X_batch, bias)      # predictions
            error = hofX - Y_batch
            gradient = (X_batch.T @ error) / batch_m + (lambda_reg / batch_m) * theta
            gradient_bias = np.mean(error)

            theta = theta - learning_rate * gradient
            bias = bias - learning_rate * gradient_bias
        
        all_preds = sigmoid(theta, X, bias)
        current_cost = -np.mean(Y * np.log(all_preds + eps) + (1 - Y) * np.log(1 - all_preds + eps)) + (lambda_reg / (2 * m)) * np.sum(theta ** 2)  # Add penalty
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_theta = np.copy(theta)
            best_bias = bias
            print(f"Epoch {epoch}: New best cost= {best_cost:.4f}")
        epoch -= 1
    print(f"Best cost: {best_cost}")
    return best_cost, best_theta, best_bias


def sigmoid(theta: np.ndarray, X: np.ndarray, bias: float | np.floating) -> np.ndarray:
    z = np.dot(X, theta) + bias
    ans = 1 / (1 + np.exp(-z))
    return ans


def maxabs_scale(X: np.ndarray, max_abs=None) -> tuple[np.ndarray, float]:
    """
    Scale each feature by its max absolute value.
    Keeps 0 as 0
    """
    if max_abs is None:
        max_abs = np.max(np.abs(X), axis=0)               # calculate per column
    max_abs_safe = np.where(max_abs == 0, 1, max_abs)     # avoid divison by 0 for all zero columns
    X_scaled = X / max_abs_safe
    return X_scaled, max_abs