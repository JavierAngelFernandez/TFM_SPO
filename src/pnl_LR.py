import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────────────────────
# BASE: Regresión Lineal con SGD y custom loss
# ──────────────────────────────────────────────────────────────
class LinearPnLBase(BaseEstimator, RegressorMixin):
    """
    Regresión lineal  ŷ = Xw + b  entrenada con SGD mini-batch
    sobre una custom loss orientada a maximizar PnL.

    Hereda de sklearn para poder usarse en pipelines y GridSearchCV.
    """

    def __init__(self, learning_rate=1e-3, epochs=200,
                 batch_size=64, l2=1e-4, random_state=42):
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.l2            = l2           # regularización L2 sobre w
        self.random_state  = random_state

    # ── Subclases deben implementar esto ──────────────────────
    def _grad(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Devuelve dL/dŷ para cada muestra del batch."""
        raise NotImplementedError

    # ── Entrenamiento ──────────────────────────────────────────
    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape

        # Inicialización pequeña para evitar saturación en tanh/sigmoid
        self.w_ = rng.normal(0, 0.01, d)
        self.b_ = 0.0
        self.loss_history_ = []

        for epoch in range(self.epochs):
            idx = rng.permutation(n)
            X_s, y_s = X[idx], y[idx]
            epoch_loss = 0.0

            for start in range(0, n, self.batch_size):
                Xb = X_s[start:start + self.batch_size]
                yb = y_s[start:start + self.batch_size]

                y_pred = Xb @ self.w_ + self.b_

                # Gradiente de la loss respecto a ŷ
                dl_dyp = self._grad(yb, y_pred)           # shape (batch,)

                # Regla de la cadena: dL/dw = Xᵀ · (dL/dŷ)
                dw = (Xb.T @ dl_dyp) / len(yb) + self.l2 * self.w_
                db = dl_dyp.mean()

                self.w_ -= self.learning_rate * dw
                self.b_ -= self.learning_rate * db

                epoch_loss += self._loss_value(yb, y_pred) * len(yb)

            self.loss_history_.append(epoch_loss / n)

        return self

    def predict(self, X):
        return X @ self.w_ + self.b_

    def pnl_score(self, X, y):
        """Métrica real: PnL medio."""
        return np.mean(y * np.sign(self.predict(X)))

    def directional_accuracy(self, X, y):
        return np.mean(np.sign(self.predict(X)) == np.sign(y))

    def _loss_value(self, y, y_pred):
        raise NotImplementedError


class TanhPnLLinear(LinearPnLBase):
    """
    L  = -tanh(y · ŷ)
    dL/dŷ = -y · sech²(y · ŷ)
    """

    def _loss_value(self, y, y_pred):
        return -np.tanh(y * y_pred).mean()

    def _grad(self, y, y_pred):
        z = y * y_pred
        sech2 = 1.0 - np.tanh(z) ** 2
        return -y * sech2


class LogSigmoidPnLLinear(LinearPnLBase):
    """
    L  = -log(σ(y · ŷ))
    dL/dŷ = -y · (1 - σ(y · ŷ))
    """

    @staticmethod
    def _sigmoid(x):
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    def _loss_value(self, y, y_pred):
        z = y * y_pred
        # log(σ(z)) numéricamente estable
        return -np.where(z >= 0,
                         -np.log1p(np.exp(-z)),
                         z - np.log1p(np.exp(z))).mean()

    def _grad(self, y, y_pred):
        z = y * y_pred
        sig = self._sigmoid(z)
        return -y * (1.0 - sig)


class ExpoPnLLinear(LinearPnLBase):
    """
    L  = -(1 - exp(-α · y · ŷ))
    dL/dŷ = -α · y · exp(-α · y · ŷ)
    """

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _loss_value(self, y, y_pred):
        return -(1.0 - np.exp(-self.alpha * y * y_pred)).mean()

    def _grad(self, y, y_pred):
        z = self.alpha * y * y_pred
        return -self.alpha * y * np.exp(-z)

