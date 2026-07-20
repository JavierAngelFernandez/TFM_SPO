import numpy as np
import pandas as pd

def format_data_to_SPO(X: pd.DataFrame, y: pd.Series):
    """Formatear los datos para SPO. X e y deben tener el mismo indice"""
    df = X.copy()
    df["tgt"] = y.copy()
    df["date_day"] = df.index.date
    df["slot"] = df.index.hour * 4 + df.index.minute // 15
    X_spo = df.pivot(index="date_day", columns="slot")
    X_spo.columns = [f"{x1}_{x2}" for x1, x2 in X_spo.columns]

    tgt_cols = [col for col in X_spo.columns if col.startswith("tgt")]
    train_cols = [col for col in X_spo.columns if col not in tgt_cols]
    return X_spo[train_cols], X_spo[tgt_cols]


class SPOPlusLinearRegression:
    def __init__(self, learning_rate=1e-3, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.B = None

    def predict(self, X):
        return X @ self.B

    def fit(self, X, y, optimizador):
        n, p = X.shape
        d = y.shape[1]

        self.B = np.zeros((p, d))

        for epoch in range(self.epochs):

            grad = np.zeros_like(self.B)
            loss = 0

            for i in range(n):

                x = X[i]
                c = y.iloc[i]

                # predicción
                c_hat = x @ self.B

                # decisión óptima con precio real
                w_real = optimizador(c)

                # decisión auxiliar
                w_aux = optimizador(2*c_hat - c)

                # gradiente (paper)
                grad += 2 * np.outer(x, (w_real - w_aux))

                # pérdida SPO+
                spo = np.dot(c - 2*c_hat, w_aux)
                spo += 2*np.dot(c_hat, w_real)
                spo -= np.dot(c, w_real)

                loss += spo
                

            grad /= n
            loss /= n

            self.B -= self.lr * grad

            print(
                f"Epoch {epoch+1:3d}   Loss={loss:.4f}"
            )


