import numpy as np, pandas as pd

df = pd.read_csv('../data/dataset_clean.csv')

y = np.log(df['price_pen'].values)  # log-precio: corrige la fuerte asimetria (skew 2.52 -> 0.11)

num_cols = ['bedrooms', 'bathrooms', 'area_m2', 'age_years', 'maintenance_pen']
X_num = df[num_cols].copy()

# Dummies (se elimina una categoria de referencia en cada grupo)
house_dum = pd.get_dummies(df['housing_type'], prefix='tipo', drop_first=True, dtype=float)
dist_dum = pd.get_dummies(df['district_grp'], prefix='distrito', drop_first=True, dtype=float)
# referencia (categoria omitida): tipo=Apartamento, distrito=Barranco

has_maint = df[['has_maintenance']].astype(float)

# Interacciones de interes: area con dummies de distrito (precio por m2 distinto
# segun zona) y area*dormitorios, area*banos (tamano de ambientes)
inter = pd.DataFrame({
    'area*bedrooms': X_num['area_m2'] * X_num['bedrooms'],
    'area*bathrooms': X_num['area_m2'] * X_num['bathrooms'],
    'bedrooms*bathrooms': X_num['bedrooms'] * X_num['bathrooms'],
})
for c in dist_dum.columns:
    inter[f'area*{c}'] = X_num['area_m2'] * dist_dum[c]

quad = pd.DataFrame({
    'area_m2^2': X_num['area_m2'] ** 2,
    'age_years^2': X_num['age_years'] ** 2,
})

Xfull = pd.concat([X_num, has_maint, house_dum, dist_dum, inter, quad], axis=1)
Xfull = Xfull.astype(float)

print("n =", Xfull.shape[0], " p =", Xfull.shape[1])
print(Xfull.columns.tolist())

Xfull.to_csv('../data/X_full.csv', index=False)
pd.Series(y, name='log_price').to_csv('../data/y.csv', index=False)
df.to_csv('../data/dataset_clean.csv', index=False)
