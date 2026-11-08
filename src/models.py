"""
Implementaciones 'from scratch' de Ridge y Elastic Net, siguiendo exactamente
las fórmulas del informe (informe_entrega1_v3.tex):
  - Ridge: solución cerrada (X^T X + lambda I)^{-1} X^T y
  - Elastic Net: descenso por coordenadas con umbralización suave,
        beta_j <- S_{lambda*alpha}(x_j^T r_j) / (||x_j||^2 + lambda*(1-alpha))
Convenciones: datos centrados y estandarizados (||x_j||^2 = n), función objetivo
SIN el factor 1/n (ver Sección "Correspondencia con scikit-learn" del informe).
"""
import numpy as np


def soft_threshold(z, tau):
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)


class RidgeScratch:
    """Ridge con solución cerrada: beta_R = (X^T X + lambda I)^-1 X^T y."""

    def __init__(self, lam, fit_intercept=True):
        self.lam = lam
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n, p = X.shape
        if self.fit_intercept:
            Xc = X - X.mean(axis=0)
            yc = y - y.mean()
        else:
            Xc, yc = X, y
        XtX = Xc.T @ Xc
        beta = np.linalg.solve(XtX + self.lam * np.eye(p), Xc.T @ yc)
        self.coef_ = beta
        self.intercept_ = (y.mean() - X.mean(axis=0) @ beta) if self.fit_intercept else 0.0
        return self

    def predict(self, X):
        return self.intercept_ + X @ self.coef_


class ElasticNetScratch:
    """Elastic Net por descenso por coordenadas (Algoritmo 1 del informe).

    beta_j <- S_{lambda*alpha}(x_j^T r_j) / (||x_j||^2 + lambda*(1-alpha))
    """

    def __init__(self, lam, alpha, tol=1e-6, max_iter=2000, fit_intercept=True):
        self.lam = lam
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        self.loss_path_ = None

    def fit(self, X, y, warm_start=None, track_loss=False):
        n, p = X.shape
        if self.fit_intercept:
            Xc = X - X.mean(axis=0)
            yc = y - y.mean()
        else:
            Xc, yc = X, y
        xj_norm2 = np.sum(Xc ** 2, axis=0)  # ||x_j||^2 por columna

        beta = np.zeros(p) if warm_start is None else warm_start.copy()
        r = yc - Xc @ beta

        loss_path = [] if track_loss else None
        lam, alpha = self.lam, self.alpha

        for it in range(1, self.max_iter + 1):
            max_change = 0.0
            for j in range(p):
                if xj_norm2[j] == 0:
                    continue
                r_j = r + Xc[:, j] * beta[j]        # residuo parcial (quitando j)
                z_j = Xc[:, j] @ r_j
                beta_new = soft_threshold(z_j, lam * alpha) / (xj_norm2[j] + lam * (1 - alpha))
                delta = beta_new - beta[j]
                if delta != 0.0:
                    r = r_j - Xc[:, j] * beta_new   # actualizar residuo completo
                    beta[j] = beta_new
                    max_change = max(max_change, abs(delta))
            if track_loss:
                loss = 0.5 * np.sum(r ** 2) + lam * (alpha * np.sum(np.abs(beta))
                                                       + 0.5 * (1 - alpha) * np.sum(beta ** 2))
                loss_path.append(loss)
            if max_change < self.tol:
                break

        self.coef_ = beta
        self.intercept_ = (y.mean() - X.mean(axis=0) @ beta) if self.fit_intercept else 0.0
        self.n_iter_ = it
        self.loss_path_ = loss_path
        return self

    def predict(self, X):
        return self.intercept_ + X @ self.coef_
