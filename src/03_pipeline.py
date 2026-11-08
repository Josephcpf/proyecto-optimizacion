import numpy as np, pandas as pd, json
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge, ElasticNet
from models import RidgeScratch, ElasticNetScratch

np.random.seed(42)

Xfull = pd.read_csv('../data/X_full.csv')
y = pd.read_csv('../data/y.csv')['log_price'].values
feat_names = Xfull.columns.tolist()
X = Xfull.values
n, p = X.shape
print("n =", n, " p =", p)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizacion SOLO con train (evita fuga de informacion)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
sigma[sigma == 0] = 1.0
X_train_s = (X_train - mu) / sigma
X_test_s = (X_test - mu) / sigma


def metrics(y_true, y_pred):
    resid = y_true - y_pred
    mse = np.mean(resid ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(resid))
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return dict(R2=r2, RMSE=rmse, MSE=mse, MAE=mae)


# ---------------------------------------------------------------
# 1) Validacion cruzada 5-fold para RIDGE
# ---------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lambdas_ridge = np.logspace(-2, 3, 25)

cv_ridge_rows = []
for lam in lambdas_ridge:
    errs = []
    for tr_idx, va_idx in kf.split(X_train_s):
        Xtr, Xva = X_train_s[tr_idx], X_train_s[va_idx]
        ytr, yva = y_train[tr_idx], y_train[va_idx]
        mu2, sd2 = Xtr.mean(0), Xtr.std(0); sd2[sd2 == 0] = 1.0
        Xtr2 = (Xtr - mu2) / sd2; Xva2 = (Xva - mu2) / sd2
        m = RidgeScratch(lam=lam).fit(Xtr2, ytr)
        pred = m.predict(Xva2)
        errs.append(np.mean((yva - pred) ** 2))
    cv_ridge_rows.append((lam, np.mean(errs), np.std(errs)))

cv_ridge = pd.DataFrame(cv_ridge_rows, columns=['lambda', 'cv_mse', 'cv_std'])
best_ridge_idx = cv_ridge['cv_mse'].idxmin()
lam_min = cv_ridge.loc[best_ridge_idx, 'lambda']
mse_min = cv_ridge.loc[best_ridge_idx, 'cv_mse']
se_min = cv_ridge.loc[best_ridge_idx, 'cv_std'] / np.sqrt(5)
within = cv_ridge[cv_ridge['cv_mse'] <= mse_min + se_min]
lam_1se = within['lambda'].max()

print("Ridge lambda_min =", lam_min, " MSE(cv)=", mse_min)
print("Ridge lambda_1se =", lam_1se)
cv_ridge.to_csv('../data/cv_ridge.csv', index=False)

# ---------------------------------------------------------------
# 2) Validacion cruzada para ELASTIC NET (grilla lambda x alpha)
# ---------------------------------------------------------------
lambdas_en = np.logspace(-1.5, 2, 14)
alphas_en = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

cv_en_rows = []
for alpha in alphas_en:
    for lam in lambdas_en:
        errs = []
        for tr_idx, va_idx in kf.split(X_train_s):
            Xtr, Xva = X_train_s[tr_idx], X_train_s[va_idx]
            ytr, yva = y_train[tr_idx], y_train[va_idx]
            mu2, sd2 = Xtr.mean(0), Xtr.std(0); sd2[sd2 == 0] = 1.0
            Xtr2 = (Xtr - mu2) / sd2; Xva2 = (Xva - mu2) / sd2
            m = ElasticNetScratch(lam=lam, alpha=alpha, tol=1e-5, max_iter=3000).fit(Xtr2, ytr)
            pred = m.predict(Xva2)
            errs.append(np.mean((yva - pred) ** 2))
        cv_en_rows.append((alpha, lam, np.mean(errs), np.std(errs)))

cv_en = pd.DataFrame(cv_en_rows, columns=['alpha', 'lambda', 'cv_mse', 'cv_std'])
best_en_idx = cv_en['cv_mse'].idxmin()
alpha_opt = cv_en.loc[best_en_idx, 'alpha']
lam_opt = cv_en.loc[best_en_idx, 'lambda']
print("EN optimo: alpha=", alpha_opt, " lambda=", lam_opt, " MSE(cv)=", cv_en.loc[best_en_idx, 'cv_mse'])
cv_en.to_csv('../data/cv_en.csv', index=False)

# ---------------------------------------------------------------
# 3) Ajuste final sobre TODO el train, evaluacion en test
# ---------------------------------------------------------------
ridge_final = RidgeScratch(lam=lam_min).fit(X_train_s, y_train)
pred_tr_r = ridge_final.predict(X_train_s)
pred_te_r = ridge_final.predict(X_test_s)
m_ridge_train = metrics(y_train, pred_tr_r)
m_ridge_test = metrics(y_test, pred_te_r)

en_final = ElasticNetScratch(lam=lam_opt, alpha=alpha_opt, tol=1e-8, max_iter=8000).fit(
    X_train_s, y_train, track_loss=True)
pred_tr_e = en_final.predict(X_train_s)
pred_te_e = en_final.predict(X_test_s)
m_en_train = metrics(y_train, pred_tr_e)
m_en_test = metrics(y_test, pred_te_e)

# OLS de referencia (lambda=0) para mostrar el problema sin regularizar
ols = RidgeScratch(lam=1e-8).fit(X_train_s, y_train)
m_ols_test = metrics(y_test, ols.predict(X_test_s))
m_ols_train = metrics(y_train, ols.predict(X_train_s))

print("OLS   train:", m_ols_train, " test:", m_ols_test)
print("Ridge train:", m_ridge_train, " test:", m_ridge_test)
print("EN iters:", en_final.n_iter_, " train:", m_en_train, " test:", m_en_test)

n_active_en = int(np.sum(np.abs(en_final.coef_) > 1e-8))
print("EN variables activas:", n_active_en, "/", p)

# ---------------------------------------------------------------
# 4) Validacion contra scikit-learn (en el optimo)
# ---------------------------------------------------------------
sk_ridge = Ridge(alpha=lam_min).fit(X_train_s, y_train)
sk_en = ElasticNet(alpha=lam_opt / len(y_train), l1_ratio=alpha_opt, fit_intercept=True,
                    tol=1e-10, max_iter=200000).fit(X_train_s, y_train)

pred_sk_r = sk_ridge.predict(X_test_s)
pred_sk_e = sk_en.predict(X_test_s)
m_sk_ridge_test = metrics(y_test, pred_sk_r)
m_sk_en_test = metrics(y_test, pred_sk_e)

diff_ridge_coef = np.max(np.abs(ridge_final.coef_ - sk_ridge.coef_))
diff_ridge_intercept = abs(ridge_final.intercept_ - sk_ridge.intercept_)
diff_en_coef = np.max(np.abs(en_final.coef_ - sk_en.coef_))
diff_en_intercept = abs(en_final.intercept_ - sk_en.intercept_)

print("Ridge vs sklearn: max|dcoef|=", diff_ridge_coef, " dintercept=", diff_ridge_intercept)
print("EN vs sklearn: max|dcoef|=", diff_en_coef, " dintercept=", diff_en_intercept)
print("Ridge sklearn test R2:", m_sk_ridge_test['R2'], " EN sklearn test R2:", m_sk_en_test['R2'])

# ---------------------------------------------------------------
# 5) Camino de regularizacion (Elastic Net, alpha_opt fijo)
# ---------------------------------------------------------------
lam_path = np.logspace(np.log10(lambdas_en.max()) + 0.5, -1.8, 60)
coef_path = []
n_active_path = []
warm = np.zeros(p)
for lam in lam_path:
    m = ElasticNetScratch(lam=lam, alpha=alpha_opt, tol=1e-6, max_iter=3000)
    m.fit(X_train_s, y_train, warm_start=warm)
    warm = m.coef_.copy()
    coef_path.append(warm.copy())
    n_active_path.append(int(np.sum(np.abs(warm) > 1e-8)))
coef_path = np.array(coef_path)

np.savez('../data/reg_path.npz', lam_path=lam_path, coef_path=coef_path,
         n_active_path=np.array(n_active_path), feat_names=np.array(feat_names))

# ---------------------------------------------------------------
# 6) Top coeficientes Elastic Net (efecto sobre log-precio -> % precio)
# ---------------------------------------------------------------
coef_en = en_final.coef_
order = np.argsort(-np.abs(coef_en))
top10 = [(feat_names[i], float(coef_en[i]), float((np.exp(coef_en[i] * (1/sigma[i])) - 1))) for i in order[:12]]

# ---------------------------------------------------------------
# Guardar todo para las figuras y el LaTeX
# ---------------------------------------------------------------
results = dict(
    n=n, p=p, n_train=len(y_train), n_test=len(y_test),
    lam_min=float(lam_min), lam_1se=float(lam_1se), mse_cv_ridge=float(mse_min),
    alpha_opt=float(alpha_opt), lam_opt=float(lam_opt), mse_cv_en=float(cv_en.loc[best_en_idx, 'cv_mse']),
    m_ols_train=m_ols_train, m_ols_test=m_ols_test,
    m_ridge_train=m_ridge_train, m_ridge_test=m_ridge_test,
    m_en_train=m_en_train, m_en_test=m_en_test,
    en_iters=int(en_final.n_iter_), en_n_active=n_active_en,
    diff_ridge_coef=float(diff_ridge_coef), diff_ridge_intercept=float(diff_ridge_intercept),
    diff_en_coef=float(diff_en_coef), diff_en_intercept=float(diff_en_intercept),
    m_sk_ridge_test=m_sk_ridge_test, m_sk_en_test=m_sk_en_test,
)
with open('../data/results.json', 'w') as f:
    json.dump(results, f, indent=2)

np.savez('../data/final_fit.npz',
         ridge_coef=ridge_final.coef_, ridge_intercept=ridge_final.intercept_,
         en_coef=en_final.coef_, en_intercept=en_final.intercept_,
         en_loss_path=np.array(en_final.loss_path_),
         pred_te_r=pred_te_r, pred_te_e=pred_te_e, y_test=y_test,
         feat_names=np.array(feat_names), sigma=sigma)

with open('../data/top_coefs.json', 'w') as f:
    json.dump(top10, f, indent=2)

print("\nOK - pipeline completo")
print("Top coefs EN:", top10[:6])
