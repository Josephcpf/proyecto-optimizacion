import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ACENTO = '#005F69'
GRIS = '#6E6E6E'
plt.rcParams.update({
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200,
})

final = np.load('../data/final_fit.npz', allow_pickle=True)
regpath = np.load('../data/reg_path.npz', allow_pickle=True)
cv_ridge = pd.read_csv('../data/cv_ridge.csv')
cv_en = pd.read_csv('../data/cv_en.csv')
with open('../data/results.json') as f:
    R = json.load(f)

feat_names = list(final['feat_names'])

# -------------------------------------------------------------
# Figura 3: predicho vs real (Ridge y Elastic Net), en test
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharex=True, sharey=True)
y_test = final['y_test']
for ax, pred, name, r2 in zip(
        axes, [final['pred_te_r'], final['pred_te_e']],
        ['Ridge', 'Elastic Net'], [R['m_ridge_test']['R2'], R['m_en_test']['R2']]):
    ax.scatter(y_test, pred, s=14, alpha=0.55, color=ACENTO, edgecolors='none')
    lims = [min(y_test.min(), pred.min()) - 0.1, max(y_test.max(), pred.max()) + 0.1]
    ax.plot(lims, lims, '--', color=GRIS, linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('log(precio) real')
    ax.set_title(f'{name}  ($R^2={r2:.3f}$)')
axes[0].set_ylabel('log(precio) predicho')
plt.tight_layout()
plt.savefig('../figs/pred_vs_real.pdf')
plt.close()

# -------------------------------------------------------------
# Figura 4: curva de validacion cruzada Ridge (MSE vs lambda)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.plot(cv_ridge['lambda'], cv_ridge['cv_mse'], color=ACENTO, linewidth=1.6)
ax.fill_between(cv_ridge['lambda'],
                 cv_ridge['cv_mse'] - cv_ridge['cv_std'] / np.sqrt(5),
                 cv_ridge['cv_mse'] + cv_ridge['cv_std'] / np.sqrt(5),
                 color=ACENTO, alpha=0.18, linewidth=0)
ax.axvline(R['lam_min'], color='crimson', linestyle=':', linewidth=1.3,
           label=rf"$\lambda_{{\min}}={R['lam_min']:.2f}$")
ax.axvline(R['lam_1se'], color=GRIS, linestyle=':', linewidth=1.3,
           label=rf"$\lambda_{{1se}}={R['lam_1se']:.2f}$")
ax.set_xscale('log')
ax.set_xlabel(r'$\lambda$ (escala log)')
ax.set_ylabel('MSE promedio (validación cruzada, 5 pliegues)')
ax.legend(frameon=False, fontsize=8.5)
plt.tight_layout()
plt.savefig('../figs/cv_ridge.pdf')
plt.close()

# -------------------------------------------------------------
# Figura 5: camino de regularizacion Elastic Net
# -------------------------------------------------------------
lam_path = regpath['lam_path']
coef_path = regpath['coef_path']
n_active_path = regpath['n_active_path']

fig, ax = plt.subplots(figsize=(6.4, 4.2))
cmap = plt.get_cmap('tab20')
# marcar solo las 10 variables de mayor |coef| final para no saturar
final_abs = np.abs(coef_path[-1])
top_idx = np.argsort(-final_abs)[:10]
for k, j in enumerate(top_idx):
    ax.plot(lam_path, coef_path[:, j], color=cmap(k % 20), linewidth=1.3,
            label=feat_names[j])
for j in range(coef_path.shape[1]):
    if j not in top_idx:
        ax.plot(lam_path, coef_path[:, j], color='0.85', linewidth=0.7, zorder=0)
ax.axvline(R['lam_opt'], color='crimson', linestyle=':', linewidth=1.3)
ax.set_xscale('log')
ax.invert_xaxis()
ax.set_xlabel(r'$\lambda$ (escala log, decreciente)')
ax.set_ylabel(r'$\hat\beta_j$ (coeficientes estandarizados)')
ax.legend(fontsize=6.3, ncol=2, frameon=False, loc='upper left')
plt.tight_layout()
plt.savefig('../figs/reg_path.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.plot(lam_path, n_active_path, color=ACENTO, linewidth=1.6, marker='o', markersize=2.5)
ax.axvline(R['lam_opt'], color='crimson', linestyle=':', linewidth=1.3,
           label=rf"$\lambda^\star={R['lam_opt']:.2f}$")
ax.set_xscale('log'); ax.invert_xaxis()
ax.set_xlabel(r'$\lambda$ (escala log, decreciente)')
ax.set_ylabel('N.º de coeficientes no nulos')
ax.legend(frameon=False, fontsize=8.5)
plt.tight_layout()
plt.savefig('../figs/n_active.pdf')
plt.close()

# -------------------------------------------------------------
# Figura 6: convergencia del descenso por coordenadas (Elastic Net)
# -------------------------------------------------------------
loss = final['en_loss_path']
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.plot(np.arange(1, len(loss) + 1), loss - loss.min() + 1e-10, color=ACENTO, linewidth=1.5)
ax.set_yscale('log')
ax.set_xlabel('Ciclo del descenso por coordenadas')
ax.set_ylabel(r'$F_E(\beta_k) - F_E^\star$ (escala log)')
plt.tight_layout()
plt.savefig('../figs/convergencia_en.pdf')
plt.close()

# -------------------------------------------------------------
# Figura 7: coeficientes Ridge vs Elastic Net (top variables)
# -------------------------------------------------------------
ridge_coef = final['ridge_coef']
en_coef = final['en_coef']
order = np.argsort(-np.abs(en_coef))[:12]
labels = [feat_names[i] for i in order]
y_pos = np.arange(len(order))

fig, ax = plt.subplots(figsize=(6.6, 4.6))
h = 0.35
ax.barh(y_pos + h/2, ridge_coef[order], height=h, color=GRIS, label='Ridge')
ax.barh(y_pos - h/2, en_coef[order], height=h, color=ACENTO, label='Elastic Net')
ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel(r'Coeficiente estandarizado $\hat\beta_j$')
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig('../figs/coef_comparacion.pdf')
plt.close()

# -------------------------------------------------------------
# Figura 8: residuos vs predicho (Elastic Net)
# -------------------------------------------------------------
resid = y_test - final['pred_te_e']
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.scatter(final['pred_te_e'], resid, s=14, alpha=0.55, color=ACENTO, edgecolors='none')
ax.axhline(0, color=GRIS, linestyle='--', linewidth=1)
ax.set_xlabel('log(precio) predicho (Elastic Net)')
ax.set_ylabel('Residuo')
plt.tight_layout()
plt.savefig('../figs/residuos.pdf')
plt.close()

print("Figuras guardadas en ../figs/")
