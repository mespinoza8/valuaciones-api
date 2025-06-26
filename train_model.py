import os
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import requests



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
from model import entrenar_y_guardar_modelo

# --- 1) Configuración de conexión a BD y rutas SHP ---
DB_URI = (
    f"mysql+pymysql://"
    f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('HOST')}:{os.getenv('PORT','3306')}/"
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
        divisa,
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
        estacionamientos,
        antiguedad,
        orientacion,
        latitud,
        longitud
    FROM witness_scrapper
""")

# --- 2) Cargar datos brutos desde MySQL ---
engine = create_engine(DB_URI)
with engine.connect() as conn:
    df = pd.read_sql(sql=query, con=conn)

# --- 3) Preprocessing idéntico al de tu API ---
# 3.1 Convertir precios a UF (asegura que 'precio' sea float para evitar warnings)
df['precio'] = df['precio'].astype(float)
df = convertir_precio(df, valor_uf=39500)

# 3.2 Limpiar columnas numéricas
for col in ['superficie_util', 'superficie_total', 'antiguedad', 'banos', 'dormitorios']:
    df[col] = df[col].apply(limpieza)

# 3.3 Normalizar nulos y rellenar a partir de 'desc'
df = preprocesar_nulos(df)
df = rellenar_estacionamientos(df)
df = rellenar_dormitorios(df)
df['antiguedad'] = df['antiguedad'].apply(lambda x: 2025 - x if x >= 1000 else x)

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

df_metrics.to_parquet('/Users/mespinoza/Documents/Projects/Modelo Final - Valuaciones/data_preprocessed/df_metrics.parquet')

df=df.drop(columns=['geometry','source','comuna','URL','disponible','fecha_creacion',
                    'fecha_modificacion','orientacion','id','name','desc','ubicacion','estacionamientos','index_right','antiguedad'],axis=1)


df_model = df[mask].copy()

# --- 4) Entrenar y guardar el modelo ---
# Ahora pasamos df_model, SHP_PATHS y la ruta del modelo
RESULTADO, MEJOR = entrenar_y_guardar_modelo(df_model)

print("Resultados CV de cada modelo:", RESULTADO)
print("Mejor modelo seleccionado:", MEJOR)
print("Archivo 'modelo_valoracion.pkl' creado correctamente.")