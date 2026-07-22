import numpy as np

def loss(y: np.ndarray, y_hat: np.ndarray) -> float | np.floating:
    """
    Calculates Log Loss
    """
    eps = 1e-15
    return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

def gradients(X: np.ndarray, y: np.ndarray, y_hat: np.ndarray, w: np.ndarray | None = None, lambda_reg: float = 0.0) -> tuple[np.ndarray, float | np.floating]:
    """
    Calculates dw and db
    Supports optional L2 regularization.
    """
    m = X.shape[0]
    error = y_hat - y    # shape (m,)
    dw = (1 / m) * np.dot(X.T, error)
    if lambda_reg > 0 and w is not None:
        dw = dw + (lambda_reg / m) * w
    db = np.mean(error)

    return dw, db

def train_full_batch(X: np.ndarray, Y: np.ndarray, epoch: int, learning_rate=0.1, lambda_reg=0.01) -> tuple[float | np.floating, np.ndarray, float | np.floating]:
    """
    Logistic Regression function using Full-batch Gradient Descent
    """
    m = X.shape[0]
    theta = np.zeros(X.shape[1])
    bias = 0.0

    best_cost = float("inf")
    best_theta = np.copy(theta)
    best_bias = bias

    while (epoch > 0):
        z = np.dot(X, theta) + bias
        hofX = sigmoid(z)      # predictions
        gradient, gradient_bias = gradients(X, Y, hofX, theta, lambda_reg)

        theta = theta - learning_rate * gradient
        bias = bias - learning_rate * gradient_bias
        
        log_loss = loss(Y, hofX)
        current_cost = log_loss + (lambda_reg / (2 * m)) * np.sum(theta ** 2)  # Penalty
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_theta = np.copy(theta)
            best_bias = bias
            print(f"Epoch {epoch}: New best cost = {best_cost:.4f}")
        epoch -= 1
    print(f"Best cost: {best_cost}")
    return best_cost, best_theta, best_bias


def train(X: np.ndarray, Y: np.ndarray, batch_size=512, epoch=5, learning_rate=0.1, seed=42, lambda_reg=0.01) -> tuple[float | np.floating, np.ndarray, float | np.floating]:
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

            z = np.dot(X_batch, theta) + bias
            hofX = sigmoid(z)      # predictions
            gradient, gradient_bias = gradients(X_batch, Y_batch, hofX, theta, lambda_reg)

            theta = theta - learning_rate * gradient
            bias = bias - learning_rate * gradient_bias
        
        z1 = np.dot(X, theta) + bias
        all_preds = sigmoid(z1)
        log_loss = loss(Y, all_preds)
        current_cost = log_loss + (lambda_reg / (2 * m)) * np.sum(theta ** 2)  # Penalty
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_theta = np.copy(theta)
            best_bias = bias
            print(f"Epoch {epoch}: New best cost= {best_cost:.4f}")
        epoch -= 1
    print(f"Best cost: {best_cost}")
    return best_cost, best_theta, best_bias


def sigmoid(z) -> np.ndarray:
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

def predict(X: np.ndarray, w: np.ndarray, b: float | np.floating, return_prob: bool = False) -> np.ndarray:
    z = np.dot(X, w) + b
    probabilities = sigmoid(z)

    if return_prob:
        return probabilities
    
    predictions = (probabilities >= 0.5).astype(int)
    return predictions