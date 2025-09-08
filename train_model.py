import os
import json
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import requests
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score



# Carga variables de entorno desde .env
load_dotenv()

from utils import (
    convertir_precio,
    limpieza,
    preprocesar_nulos,
    rellenar_estacionamientos,
    rellenar_dormitorios,
    geometry_points,
    calculate_nearest_distances,
    calculate_nearest_distances_metro
)
from model import preparar_datos_para_modelo, models

# --- 1) Configuración de conexión a BD y rutas SHP ---
DB_URI = (
    f"mysql+pymysql://"
    f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('PORT','3306')}/"
    f"{os.getenv('DATABASE','ml_valoranet')}"
)
SHP_PATHS = {
    'ed_superior': os.environ['ED_SUPERIOR_SHP'],
    'ed_escolar':  os.environ['ED_ESCOLAR_SHP'],
    'comisarias':  os.environ['COMISARIAS_SHP'],
    'salud':       os.environ['SALUD_SHP'],
    'metro':       os.environ['METRO_SHP'],
    'comunas':     os.environ['COMUNAS_SHP']
}

query = text("""
    SELECT
        id,
        name,
        URL,
        case when divisa='CLP' then '$'
        when divisa='CLF' then 'UF' 
        when divisa='US$' then 'US$'else divisa end as divisa,
        precio,
        `desc`,
        ubicacion,
        source,
        disponible,
        fecha_creacion,
        fecha_modificacion,
        tipo,
        comuna,
        superficie_total,
        superficie_util,
        dormitorios,
        banos,
        case when estacionamientos>0 then TRUE else FALSE end as estacionamientos,
        antiguedad,
        orientacion,
        latitud,
        longitud
    FROM witness_scrapper
    where divisa in ('UF',
'$',
'CLP',
'CLF',
'US$'
    )
""")

# --- 2) Cargar datos brutos desde MySQL ---
engine = create_engine(DB_URI)
with engine.connect() as conn:
    df = pd.read_sql(sql=query, con=conn)

# --- 2.1) Cargar datos validados desde la API
resultados_api=pd.read_excel('resultados_qa.xlsx')
resultados_api = resultados_api.fillna('')

df=pd.concat([df, resultados_api], ignore_index=True)


# --- 3) Preprocessing idéntico al de tu API ---
# 3.1 Convertir precios a UF (asegura que 'precio' sea float para evitar warnings)
df['precio'] = df['precio'].astype(float)
df = convertir_precio(df, valor_uf=39200)

# 3.2 Limpiar columnas numéricas
for col in ['superficie_util', 'superficie_total', 'antiguedad', 'banos', 'dormitorios']:
    df[col] = df[col].apply(limpieza)

# 3.3 Normalizar nulos y rellenar a partir de 'desc'
df = preprocesar_nulos(df)
df = rellenar_estacionamientos(df)
df = rellenar_dormitorios(df)
df['antiguedad'] = df['antiguedad'].apply(lambda x: 2025 - x if x >= 1000 else x)
df['estacionamientos'] = df['estacionamientos'].apply(lambda x: True if x>0 else False)

# 3.4 Calcular distancias geoespaciales
gp = geometry_points(df)
ed_sup = gpd.read_parquet(SHP_PATHS['ed_superior'])
ed_esc = gpd.read_parquet(SHP_PATHS['ed_escolar'])
comi   = gpd.read_parquet(SHP_PATHS['comisarias'])
salud  = gpd.read_parquet(SHP_PATHS['salud'])
metro  = gpd.read_parquet(SHP_PATHS['metro'])

comunas_gdf = gpd.read_parquet(SHP_PATHS['comunas'])


comunas_gdf = comunas_gdf.to_crs(gp.crs)

df = gpd.sjoin(gp, comunas_gdf[['geometry', 'Comuna','Region']], how="left")


df['distancia_ed_superior_km'] = calculate_nearest_distances(gp, ed_sup)
df['distancia_ed_escolar_km']  = calculate_nearest_distances(gp, ed_esc)
df['distancia_comisaria_km']   = calculate_nearest_distances(gp, comi)
df['distancia_est_salud_km']   = calculate_nearest_distances(gp, salud)
df['distancia_metro_km']       = calculate_nearest_distances_metro(gp, metro)

# 3.5 Filtrar outliers (misma máscara que en tu script original)
mask = (
    (df['dormitorios'] > 0) & (df['dormitorios'] < 15) &
    (df['banos']       > 0) & (df['banos']       < 10) &
    (df['superficie_total'] > 0) & (df['superficie_total'] < 20000) &
    (df['superficie_util']  > 0) & (df['superficie_util']  < 20000) &
    (df['precio']      > 0) & (df['precio']      < 25000)
)

df_metrics = df[mask].copy()

df_metrics.to_parquet('data_preprocessed/df_metrics.parquet')

df=df.drop(columns=['geometry','source','comuna','URL','disponible','fecha_creacion',
                    'fecha_modificacion','orientacion','id','name','desc','ubicacion','index_right','divisa','antiguedad'],axis=1)

#estacionamientos

df_model = df[mask].copy()

# --- 4) Ajuste de hiperparámetros con GridSearchCV y guardado ---

# 4.1 Preparar matrices y preprocesador reutilizando la lógica de model.py
X_train, X_test, y_train, y_test, preproc = preparar_datos_para_modelo(df_model)
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])

# 4.2 Definir rejillas de hiperparámetros por modelo
param_grids = {
    'LightGBM': {
        'model__n_estimators': [200, 500],
        'model__learning_rate': [0.05, 0.1],
        'model__num_leaves': [31, 63],
        'model__max_depth': [-1, 10],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0],
    },
    'CatBoost': {
        'model__n_estimators': [300, 600],
        'model__depth': [6, 8],
        'model__learning_rate': [0.05, 0.1],
        'model__l2_leaf_reg': [1, 3, 5],
    },
    'Random Forest': {
        'model__n_estimators': [300, 600],
        'model__max_depth': [None, 20, 40],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2],
        'model__max_features': ['sqrt', 'log2', None],
    }
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

resultados = {}
mejores_est = {}

from sklearn.pipeline import Pipeline

for nombre_modelo, estimador in models.items():
    pipe = Pipeline([
        ('preprocessor', preproc),
        ('model', estimador)
    ])

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grids.get(nombre_modelo, {}),
        scoring='neg_root_mean_squared_error',
        cv=kf,
        n_jobs=-1,
        verbose=0,
        refit=True,
        return_train_score=False
    )

    grid.fit(X_full, y_full)
    best_estimator = grid.best_estimator_
    mejores_est[nombre_modelo] = best_estimator

    # Métricas consistentes con el modelo anterior (CV sobre todo el dataset)
    y_pred = cross_val_predict(best_estimator, X_full, y_full, cv=kf, n_jobs=-1)
    mse = mean_squared_error(y_full, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_full, y_pred))
    mape = float(np.mean(np.abs((y_full - y_pred) / y_full)) * 100)

    resultados[nombre_modelo] = {
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

# 4.3 Seleccionar el mejor modelo por RMSE y entrenar final
mejor_modelo = min(resultados, key=lambda m: resultados[m]['rmse'])
modelo_final = mejores_est[mejor_modelo]
modelo_final.fit(X_full, y_full)

# 4.4 Guardar artefactos: modelo, métricas e importancias
project_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_dir, 'modelo_valoracion.pkl')
joblib.dump(modelo_final, model_path)

metrics_output = {
    'model_name': mejor_modelo,
    'metrics': resultados
}
metrics_path = os.path.join(project_dir, 'metrics.json')
with open(metrics_path, 'w', encoding='utf-8') as mf:
    json.dump(metrics_output, mf, indent=4, ensure_ascii=False)

# Importancia de variables
try:
    # Reconstruir nombres de variables tras preprocesamiento
    num_cols = X_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_full.select_dtypes(include=['object', 'category']).columns.tolist()
    preproc_fitted = modelo_final.named_steps['preprocessor']
    ohe = preproc_fitted.named_transformers_['cat'].named_steps['onehot']
    cat_features = ohe.get_feature_names_out(cat_cols)
    feature_names = list(num_cols) + list(cat_features)
except Exception:
    feature_names = None

importancias = None
try:
    importancias = modelo_final.named_steps['model'].feature_importances_
except Exception:
    importancias = None

if importancias is not None:
    if feature_names is None:
        feature_names = [f"var_{i}" for i in range(len(importancias))]
    importancia_dict = dict(sorted(zip(feature_names, importancias), key=lambda x: x[1], reverse=True))
    importancia_path = os.path.join(project_dir, 'importancia_variables.json')
    with open(importancia_path, 'w', encoding='utf-8') as f:
        json.dump(importancia_dict, f, indent=4, ensure_ascii=False)

print("Resultados CV de cada modelo:", resultados)
print("Mejor modelo seleccionado:", mejor_modelo)
print("Archivo 'modelo_valoracion.pkl' creado correctamente.")