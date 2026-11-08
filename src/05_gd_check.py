import numpy as np, pandas as pd, json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from models import RidgeScratch

np.random.seed(42)
Xfull = pd.read_csv('../data/X_full.csv')
y = pd.read_csv('../data/y.csv')['log_price'].values
X = Xfull.values
n, p = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0); sigma[sigma == 0] = 1.0
X_train_s = (X_train - mu) / sigma
X_test_s = (X_test - mu) / sigma

with open('../data/results.json') as f:
    R = json.load(f)
lam_min = R['lam_min']

# Solucion cerrada (referencia)
ridge_closed = RidgeScratch(lam=lam_min).fit(X_train_s, y_train)

# Descenso del gradiente manual (misma formula que en el informe, ec. 4)
Xc = X_train_s - X_train_s.mean(axis=0)
yc = y_train - y_train.mean()
XtX = Xc.T @ Xc
eigvals = np.linalg.eigvalsh(XtX)
s1_sq, sp_sq = eigvals.max(), eigvals.min()
L = s1_sq + lam_min
mu_cvx = sp_sq + lam_min
kappa = L / mu_cvx
eta = 1.0 / L

beta = np.zeros(p)
tol = 1e-10
max_iter = 200000
history = []
k = 0
for k in range(1, max_iter + 1):
    grad = Xc.T @ (Xc @ beta - yc) + lam_min * beta
    beta_new = beta - eta * grad
    diff = np.linalg.norm(beta_new - beta)
    beta = beta_new
    if k % 20 == 0 or k == 1:
        history.append((k, np.linalg.norm(beta - ridge_closed.coef_)))
    if diff < tol:
        break

n_iter_gd = k
diff_gd_closed = float(np.max(np.abs(beta - ridge_closed.coef_)))
print("Ridge GD: L=", L, " mu=", mu_cvx, " kappa=", kappa, " eta=", eta)
print("Ridge GD iteraciones hasta convergencia:", n_iter_gd)
print("Ridge GD vs cerrada, diff max coef:", diff_gd_closed)

sk_ridge = Ridge(alpha=lam_min).fit(X_train_s, y_train)
diff_gd_sklearn = float(np.max(np.abs(beta - sk_ridge.coef_)))
print("Ridge GD vs sklearn, diff max coef:", diff_gd_sklearn)

theor_rate = (1 - 1 / kappa) ** n_iter_gd
print("cota teorica (1-1/kappa)^k en la iteracion final:", theor_rate)

gd_results = dict(
    L=float(L), mu=float(mu_cvx), kappa=float(kappa), eta=float(eta),
    n_iter_gd=int(n_iter_gd), diff_gd_closed=diff_gd_closed, diff_gd_sklearn=diff_gd_sklearn,
    theor_rate=float(theor_rate),
)
with open('../data/gd_results.json', 'w') as f:
    json.dump(gd_results, f, indent=2)

np.savez('../data/gd_check.npz', history=np.array(history))
print("\nOK")
