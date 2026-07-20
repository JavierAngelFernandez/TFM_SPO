import numpy as np
from catboost import CatBoostRegressor, Pool


class TanhPnLLoss:
    """
    L = -tanh(y * y_pred)
    Gradiente:  -y * sech²(y * y_pred)
    Hessiana:    2*y² * tanh(z) * sech²(z)   (puede ser negativa → se clampea)
    """

    def calc_ders_range(self, approxes, targets, weights):
        result = []
        for approx, target in zip(approxes, targets):
            z = target * approx
            tanh_z = np.tanh(z)
            sech2_z = 1.0 - tanh_z ** 2

            grad = -target * sech2_z
            hess = 2.0 * (target ** 2) * tanh_z * sech2_z
            hess = max(abs(hess), 1e-6)

            result.append((grad, hess))
        return result


class LogSigmoidPnLLoss:
    """
    L = -log(σ(y * y_pred))
    Gradiente:  -y * (1 - σ(z))
    Hessiana:    y² * σ(z) * (1 - σ(z))
    """

    @staticmethod
    def _sigmoid(x):
        # Numericamente estable
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    def calc_ders_range(self, approxes, targets, weights):
        result = []
        for approx, target in zip(approxes, targets):
            z = target * approx
            sig = self._sigmoid(z)

            grad = -target * (1.0 - sig)
            hess = max((target ** 2) * sig * (1.0 - sig), 1e-6)

            result.append((grad, hess))
        return result


class ExpoPnLLoss:
    """
    L = -(1 - exp(-α * y * y_pred))
    Gradiente:  -α * y * exp(-α * y * y_pred)
    Hessiana:    α² * y² * exp(-α * y * y_pred)
    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def calc_ders_range(self, approxes, targets, weights):
        result = []
        for approx, target in zip(approxes, targets):
            z = self.alpha * target * approx
            exp_z = np.exp(-z)                     # exp(-α*y*ŷ)

            grad = -self.alpha * target * exp_z
            hess = max((self.alpha ** 2) * (target ** 2) * exp_z, 1e-6)

            result.append((grad, hess))
        return result


class PnLMetric:
    def evaluate(self, approxes, target, weight):
        preds = np.array(approxes[0])
        target = np.array(target)

        pnl = target * np.sign(preds)

        if weight is None:
            return pnl.sum(), len(target)
        else:
            weight = np.array(weight)
            return (pnl * weight).sum(), weight.sum()

    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def get_description(self):
        return "PnLMetric"


