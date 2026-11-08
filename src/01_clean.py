import re
import numpy as np
import pandas as pd

RAW = '../data/dataset_raw.csv'
TASA_USD_PEN = 3.80  # tipo de cambio aprox. usado para unificar precios en soles (feb-mar 2023)

df = pd.read_csv(RAW)
print("Filas originales:", len(df))

# -----------------------------------------------------------------
# 1) Precio -> soles (float)
# -----------------------------------------------------------------
def parse_price(s):
    s = str(s).strip()
    is_usd = s.startswith('USD')
    digits = re.sub(r'[^\d]', '', s.replace(',', ''))
    if digits == '':
        return np.nan
    val = float(digits)
    return val * TASA_USD_PEN if is_usd else val

df['price_pen'] = df['price'].apply(parse_price)

# -----------------------------------------------------------------
# 2) Dormitorios / baños -> entero
# -----------------------------------------------------------------
def parse_count(s):
    if pd.isna(s):
        return np.nan
    m = re.match(r'(\d+)', str(s))
    return float(m.group(1)) if m else np.nan

df['bedrooms'] = df['bedroom'].apply(parse_count)
df['bathrooms'] = df['bathroom'].apply(parse_count)

# -----------------------------------------------------------------
# 3) Area (m^2) -> float
# -----------------------------------------------------------------
def parse_area(s):
    if pd.isna(s):
        return np.nan
    m = re.match(r'([\d.,]+)\s*m', str(s))
    if not m:
        return np.nan
    return float(m.group(1).replace(',', ''))

df['area_m2'] = df['area'].apply(parse_area)

# -----------------------------------------------------------------
# 4) Año de construcción -> edad del inmueble (algunas filas tienen
#    'NN m²' por error de captura en esa columna; se descartan)
# -----------------------------------------------------------------
def parse_year(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    if not re.match(r'^\d{4}$', s):
        return np.nan
    y = int(s)
    return y if 1900 <= y <= 2023 else np.nan

df['year_built'] = df['year_contruction'].apply(parse_year)
df['age_years'] = 2023 - df['year_built']

# -----------------------------------------------------------------
# 5) Mantenimiento -> soles
# -----------------------------------------------------------------
def parse_maint(s):
    if pd.isna(s):
        return np.nan
    digits = re.sub(r'[^\d]', '', str(s).replace(',', ''))
    return float(digits) if digits else np.nan

df['maintenance_pen'] = df['maintenance'].apply(parse_maint)
df['has_maintenance'] = df['maintenance_pen'].notna().astype(int)
df['maintenance_pen'] = df['maintenance_pen'].fillna(0.0)

# -----------------------------------------------------------------
# 6) Distrito y departamento desde 'location'
#    Formato típico: "[Sub-zona,] Distrito, Provincia, Departamento"
# -----------------------------------------------------------------
def parse_location(s):
    parts = [p.strip() for p in str(s).split(',') if p.strip()]
    if len(parts) >= 3:
        distrito = parts[-3]
        depto = parts[-1]
    elif len(parts) == 2:
        distrito = parts[0]
        depto = parts[-1]
    else:
        distrito, depto = np.nan, np.nan
    return pd.Series([distrito, depto])

df[['district', 'department']] = df['location'].apply(parse_location)
df['district'] = df['district'].str.strip()
df['department'] = df['department'].str.strip()

# -----------------------------------------------------------------
# 7) Filtro de calidad: Apartamento/Casa, precio y área razonables,
#    y departamento = Lima (para evitar mezclar mercados muy distintos
#    con pocas observaciones cada uno)
# -----------------------------------------------------------------
df = df[df['housing_type'].isin(['Apartamento', 'Casa'])]
df = df[df['department'] == 'Lima']
df = df.dropna(subset=['price_pen', 'bedrooms', 'bathrooms', 'area_m2'])
df = df[(df['price_pen'].between(300, 30000)) & (df['area_m2'].between(15, 1000))]
df = df[(df['bedrooms'] <= 10) & (df['bathrooms'] <= 10)]

print("Filas tras limpieza:", len(df))
print(df['district'].value_counts().head(15))

# -----------------------------------------------------------------
# 8) Distritos frecuentes como variables dummy (resto -> 'Otro')
# -----------------------------------------------------------------
top_districts = df['district'].value_counts()
top_districts = top_districts[top_districts >= 15].index.tolist()
df['district_grp'] = df['district'].where(df['district'].isin(top_districts), 'Otro')
print("\nDistritos usados como dummy:", top_districts)

df_clean = df[['price_pen', 'bedrooms', 'bathrooms', 'area_m2', 'age_years',
                'maintenance_pen', 'has_maintenance', 'housing_type',
                'district_grp']].copy()
df_clean['age_years'] = df_clean['age_years'].fillna(df_clean['age_years'].median())
df_clean.to_csv('../data/dataset_clean.csv', index=False)
print("\nGuardado data/dataset_clean.csv:", df_clean.shape)
print(df_clean.describe())
