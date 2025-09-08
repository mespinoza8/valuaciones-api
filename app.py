import os
from dotenv import load_dotenv
import pandas as pd
import geopandas as gpd
from flask import Flask, request, jsonify
from utils import create_point_gdf, calculate_nearest_distances, calculate_nearest_distances_metro
import joblib
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from datetime import datetime
from flasgger import Swagger
import subprocess
import jwt
import json
import unicodedata
import numpy as np 
from data_metrics import metricas_comuna

load_dotenv()

# Utility: Get comuna from coordinates using spatial join
def get_comuna_from_coords(lat, lon, comunas_gdf):
    punto = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([lon], [lat]),
        crs=comunas_gdf.crs
    )
    join = gpd.sjoin(punto, comunas_gdf, how='left', predicate='within')
    if join.empty or pd.isnull(join.iloc[0]['Comuna']):
        raise ValueError("Coordenadas fuera de los límites de las comunas conocidas.")
    return join.iloc[0]['Comuna']


def normalize_str(s: str) -> str:
    nfkd = unicodedata.normalize('NFD', s)
    only_base = ''.join(ch for ch in nfkd if unicodedata.category(ch) != 'Mn')
    return only_base.lower().strip()


# Configuración

SECRET_KEY = os.getenv('SECRET_KEY')
MODEL_PATH = os.getenv('MODEL_PATH', 'modelo_valoracion.pkl')
SHP_PATHS = {
    'ed_superior': os.getenv('ED_SUPERIOR_SHP', 'data_preprocessed/ed_superior.parquet').strip("'\""),
    'ed_escolar':  os.getenv('ED_ESCOLAR_SHP',  'data_preprocessed/ed_escolar.parquet').strip("'\""),
    'comisarias':  os.getenv('COMISARIAS_SHP',  'data_preprocessed/comisarias.parquet').strip("'\""),
    'salud':       os.getenv('SALUD_SHP',       'data_preprocessed/salud.parquet').strip("'\""),
    'metro':       os.getenv('METRO_SHP',       'data_preprocessed/metro.parquet').strip("'\""),
    'comunas':     os.getenv('COMUNAS_SHP',     'data_preprocessed/comunas.parquet').strip("'\"")
}

# carga archivo de comunas y regiones
MAPPING_FILE = os.getenv('COMUNA_REGION_FILE', 'comunas.xlsx')
map_df = pd.read_excel(MAPPING_FILE)
map_df.columns = map_df.columns.str.strip()
raw_map = dict(zip(map_df['Comuna'], map_df['Region']))

COMUNA_REGION_MAP = {
    normalize_str(comuna_name): region
    for comuna_name, region in raw_map.items()
}

# --- Configuración de la Base de Datos MariaDB ---
db_user = os.getenv('DB_USER')
db_pass = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT', '3306')
db_name = os.getenv('DB_NAME', 'ml_valoranet')
# Añadimos charset utf8mb4 para compatibilidad de caracteres
DB_URI = (
    f"mysql+pymysql://{db_user}:{db_pass}"
    f"@{db_host}:{db_port}/{db_name}"
)
engine = create_engine(
    DB_URI,
    poolclass=NullPool,
    connect_args={"connect_timeout": 10})


# Cargar el modelo serializado
model = joblib.load(MODEL_PATH)

# Cargar shapefiles en memoria
ed_sup = gpd.read_parquet(SHP_PATHS['ed_superior'])
ed_esc = gpd.read_parquet(SHP_PATHS['ed_escolar'])
comi   = gpd.read_parquet(SHP_PATHS['comisarias'])
salud  = gpd.read_parquet(SHP_PATHS['salud'])
metro  = gpd.read_parquet(SHP_PATHS['metro'])
comuna_file= gpd.read_parquet(SHP_PATHS['comunas']).to_crs(epsg=4326)



# Inicializar Flask
app = Flask(__name__)
swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "API Valuaciones",
        "description": "Estimación de valor de propiedades en UF",
        "version": "1.0.0"
    }
})

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json(force=True)
        lat = data.get('latitud')
        lon = data.get('longitud')
        if lat is None or lon is None:
            return jsonify({'error': "Debes enviar las coordenadas 'latitud' y 'longitud'"}), 400

        raw = data.get('Comuna') or data.get('comuna')
        if raw:
            comuna = normalize_str(raw)
        else:
            try:
                fallback = get_comuna_from_coords(lat, lon, comuna_file).strip().lower()
                comuna = normalize_str(fallback)
            except ValueError as ve:
                return jsonify({'error': str(ve)}), 400

        # 3) Validar región
        region = COMUNA_REGION_MAP.get(comuna)
        if not region:
            return jsonify({'error': f"Comuna '{comuna}' no está permitida"}), 400

        # 4) Armar payload unificado
        features = {**data, 'Comuna': comuna, 'Region': region, 'divisa': 'UF'}

        # 5) Calcular distancias
        punto = create_point_gdf(lat, lon)
        features['distancia_ed_superior_km'] = calculate_nearest_distances(punto, ed_sup)[0]
        features['distancia_ed_escolar_km']  = calculate_nearest_distances(punto, ed_esc)[0]
        features['distancia_comisaria_km']   = calculate_nearest_distances(punto, comi)[0]
        features['distancia_est_salud_km']   = calculate_nearest_distances(punto, salud)[0]
        features['distancia_metro_km']       = calculate_nearest_distances_metro(punto, metro)[0]

        # 6) Predecir
        df_new = pd.DataFrame([features])
        prediction = model.predict(df_new)[0]

        # 7) Métricas por comuna
        metricas_comuna_df = metricas_comuna()
        metricas_comuna_df['comuna_norm'] = metricas_comuna_df['Comuna'].apply(normalize_str)
        cm = metricas_comuna_df[metricas_comuna_df['comuna_norm'] == comuna]

        if cm.empty:
            avg_price_uf          = 0.0
            superficie_util_prom  = 0.0
            nro_propiedades       = 0
        else:
          comuna_metrics       = cm.iloc[0]
          avg_price_uf            = comuna_metrics['avg_price_uf']
          superficie_util_prom    = comuna_metrics['superficie']
          nro_propiedades         = comuna_metrics['n_properties']

        # 8) Guardar en BD
        record = {
            **features,
            'antiguedad': 0,
            'prediction_uf':       float(prediction),
            'avg_price_uf':        float(avg_price_uf),
            'avg_price_uf_m2':     float(superficie_util_prom),
            'n_properties':        int(nro_propiedades),
            'requested_at':        datetime.now()
        }
        # insert_sql = text("""
        #     INSERT INTO model_predictions (
        #         divisa, tipo, superficie_util, superficie_total,
        #         antiguedad, dormitorios, banos,
        #         comuna, region, latitud, longitud,
        #         distancia_ed_superior_km, distancia_ed_escolar_km,
        #         distancia_comisaria_km, distancia_est_salud_km,
        #         distancia_metro_km, prediction_uf,
        #         avg_price_uf, avg_price_uf_m2, n_properties, requested_at
        #     ) VALUES (
        #         :divisa, :tipo, :superficie_util, :superficie_total,
        #         :antiguedad, :dormitorios, :banos,
        #         :Comuna, :Region, :latitud, :longitud,
        #         :distancia_ed_superior_km, :distancia_ed_escolar_km,
        #         :distancia_comisaria_km, :distancia_est_salud_km,
        #         :distancia_metro_km, :prediction_uf,
        #         :avg_price_uf, :avg_price_uf_m2, :n_properties, :requested_at
        #     )
        # """)
        # with engine.begin() as conn:
        #     conn.execute(insert_sql, record)

        # 9) Respuesta JSON
        return jsonify({
            'prediction_uf':                    float(prediction),
            'comuna':                           comuna,
            'valor_promedio_propiedades_comuna': float(avg_price_uf),
            'superficie_util_promedio_comuna':   float(superficie_util_prom),
            'cantidad_propiedades_comuna':       int(nro_propiedades)
        }), 200

    except Exception as e:
        app.logger.error(f"Error in predict_endpoint: {e}")
        return jsonify({'error': str(e)}), 400
        


@app.route('/retrain', methods=['POST'])
def retrain_endpoint():
    # Token simple como query param o header
    token = request.args.get('token') or request.headers.get('X-API-Token')
    if token != SECRET_KEY:
        return jsonify({'error': 'Invalid token'}), 401
    
    # Lanzar entrenamiento (nota: esto bloqueará la petición)
    proc = subprocess.run(['python3', 'train_model.py'], capture_output=True, text=True)
    if proc.returncode != 0:
        return jsonify({'error': proc.stderr}), 500
    
    # Recargar modelo
    global model
    model = joblib.load(MODEL_PATH)
    
    return jsonify({'status': 'success', 'message': 'Model retrained'}), 200

@app.route('/metrics', methods=['GET'])
def metrics_endpoint():
    """
    Devuelve las métricas del mejor modelo entrenado.
    ---
    tags:
      - Valuaciones
    responses:
      200:
        description: Métricas del modelo
        schema:
          type: object
          properties:
            model_name:
              type: string
            metrics:
              type: object
    """
    try:
        # Cargar métricas desde archivo JSON
        with open('metrics.json', 'r') as f:
            data = json.load(f)
        return jsonify(data), 200
    except FileNotFoundError:
        return jsonify({'error': 'metrics.json no encontrado'}), 404
    except Exception as e:
        app.logger.error(f"Error en metrics_endpoint: {e}")
        return jsonify({'error': str(e)}), 500


################################################################################################################
################################################################################################################
########################Encargos################################################################################

def cargar_modelo_v2():
    try:
        modelo = joblib.load('encargos_data/modelo_valuacion_propiedades_v2.pkl')
        scaler = joblib.load('encargos_data/scaler_valuacion_propiedades_v2.pkl')
        label_encoders = joblib.load('encargos_data/label_encoders_valuacion_propiedades_v2.pkl')
        available_features = joblib.load('encargos_data/features_valuacion_propiedades_v2.pkl')
        kmeans_model = joblib.load('encargos_data/kmeans_zona_geografica_v2.pkl')
        densidad_por_zona = joblib.load('encargos_data/densidad_por_zona_v2.pkl')
        
        return modelo, scaler, label_encoders, available_features, kmeans_model, densidad_por_zona
    except Exception as e:
        app.logger.error(f"Error cargando modelo v2: {e}")
        return None, None, None, None, None, None

def preparar_datos_entrada_v2(datos_entrada, label_encoders, available_features, kmeans_model, densidad_por_zona):
    """Prepara los datos de entrada para el modelo v2"""
    df = pd.DataFrame([datos_entrada])

    # Mapear latitud/longitud a latitud_corregida/longitud_corregida
    df['latitud_corregida'] = df['latitud']
    df['longitud_corregida'] = df['longitud']

    # Calcular densidad de construcción
    df['densidad_construccion'] = df['sup_edificada'] / df['sup_terreno'].replace(0, np.nan)
    df['densidad_construccion'] = df['densidad_construccion'].fillna(1)
    
    # Calcular antigüedad
    df['antiguedad'] = datetime.now().year - df['ano_construccion']
    
    # Crear bins de antigüedad
    df['antiguedad_bin'] = pd.cut(df['antiguedad'], 
                                  bins=[0, 10, 20, 30, 50, 100], 
                                  labels=['0-10', '11-20', '21-30', '31-50', '50+'])
    
    # Crear bins de superficie edificada
    df['sup_edificada_bin'] = pd.cut(df['sup_edificada'], 
                                     bins=[0, 50, 80, 120, 200, 1000], 
                                     labels=['0-50', '51-80', '81-120', '121-200', '200+'])
    
    # Clustering usando el modelo entrenado
    if kmeans_model is not None:
        coords = df[['latitud_corregida', 'longitud_corregida']].values
        df['zona_geografica'] = kmeans_model.predict(coords)
    else:
        df['zona_geografica'] = 0  # Valor por defecto
    
    # Calcular distancia al centro
    centro_lat, centro_lon = -33.4489, -70.6693
    df['distancia_centro'] = np.sqrt(
        (df['latitud_corregida'] - centro_lat)**2 + 
        (df['longitud_corregida'] - centro_lon)**2
    ) * 111
    
    # Mapear densidad por zona
    if densidad_por_zona is not None:
        df['densidad_zona'] = df['zona_geografica'].map(densidad_por_zona)
    else:
        df['densidad_zona'] = 0.1  # Valor por defecto
    
    # Codificar variables categóricas
    categorical_cols = ['desc_tipo_bien', 'regularizado', 'antiguedad_bin', 'sup_edificada_bin','nombre_entidad']
    
    for col in categorical_cols:
        if col in df.columns and col in label_encoders:
            try:
                df[f'{col}_encoded'] = label_encoders[col].transform(df[col].astype(str))
            except:
                # Si el valor no existe en el encoder, usar 0
                df[f'{col}_encoded'] = 0
    
    # Asegurar que todas las características requeridas estén presentes
    for feature in available_features:
        if feature not in df.columns:
            if feature == 'antiguedad':
                df[feature] = datetime.now().year - df['ano_construccion']
            else:
                df[feature] = 0
    
    X = df[available_features].fillna(0)
    
    return X

def predecir_valor_propiedad_v2(datos_propiedad, modelo, scaler, label_encoders, available_features, kmeans_model, densidad_por_zona):
    """Realiza la predicción usando el modelo v2"""
    try:
        X_prepared = preparar_datos_entrada_v2(datos_propiedad, label_encoders, available_features, kmeans_model, densidad_por_zona)
        X_scaled = scaler.transform(X_prepared)
        valor_predicho = modelo.predict(X_scaled)[0]
        return valor_predicho
    except Exception as e:
        app.logger.error(f"Error en la predicción v2: {e}")
        return None




modelo_v2, scaler_v2, label_encoders_v2, available_features_v2, kmeans_model_v2, densidad_por_zona_v2 = cargar_modelo_v2()

@app.route('/predict_encargo', methods=['POST'])
def predict_encargo_endpoint():
 
    try:
        data = request.get_json(force=True)
        
        # Validar campos requeridos
        campos_requeridos = [
            'latitud', 'longitud', 'ano_construccion',
            'sup_edificada', 'sup_terreno', 'desc_tipo_bien', 'regularizado', 'nombre_entidad'
        ]
        
        for campo in campos_requeridos:
            if campo not in data:
                return jsonify({'error': f"Campo requerido faltante: {campo}"}), 400
        
        # Validar tipos de datos
        try:
            datos_propiedad = {
                'latitud': float(data['latitud']),
                'longitud': float(data['longitud']),
                'ano_construccion': int(data['ano_construccion']),
                'sup_edificada': float(data['sup_edificada']),
                'sup_terreno': float(data['sup_terreno']),
                'desc_tipo_bien': str(data['desc_tipo_bien']),
                'regularizado': str(data['regularizado']),
                'nombre_entidad': str(data['nombre_entidad']),
            }
        except (ValueError, TypeError) as e:
            return jsonify({'error': f"Error en tipos de datos: {str(e)}"}), 400
        
        # Validar valores específicos
        if datos_propiedad['desc_tipo_bien'] not in ['Casa', 'Departamento', 'Oficina']:
            return jsonify({'error': "desc_tipo_bien debe ser 'Casa', 'Departamento' u 'Oficina'"}), 400
        
        # Normalizar y validar regularizado
        regularizado_input = datos_propiedad['regularizado'].lower().strip()
        valores_si = ['sí', 'si', 's', 'yes', 'y', 'true', '1']
        valores_no = ['no', 'n', 'false', '0']
        
        if regularizado_input in valores_si:
            datos_propiedad['regularizado'] = 'Sí'
        elif regularizado_input in valores_no:
            datos_propiedad['regularizado'] = 'No'
        else:
            return jsonify({'error': "regularizado debe ser 'Sí', 'Si', 'No' o variaciones similares"}), 400
        
        if datos_propiedad['sup_terreno'] <= 0:
            return jsonify({'error': "sup_terreno debe ser mayor a 0"}), 400
        
        if datos_propiedad['sup_edificada'] <= 0:
            return jsonify({'error': "sup_edificada debe ser mayor a 0"}), 400
        
        # Verificar que el modelo esté cargado
        if modelo_v2 is None:
            return jsonify({'error': "Modelo v2 no disponible"}), 500
        
        # Realizar predicción
        valor_predicho = predecir_valor_propiedad_v2(
            datos_propiedad, modelo_v2, scaler_v2, label_encoders_v2, 
            available_features_v2, kmeans_model_v2, densidad_por_zona_v2
        )
        
        if valor_predicho is None:
            return jsonify({'error': "No se pudo realizar la predicción"}), 500
        
        # Calcular datos adicionales para la respuesta
        antiguedad = datetime.now().year - datos_propiedad['ano_construccion']
        densidad_construccion = datos_propiedad['sup_edificada'] / datos_propiedad['sup_terreno']
        
        # Respuesta JSON
        return jsonify({
            'prediction_uf': float(valor_predicho),
            'datos_procesados': {
                'antiguedad': antiguedad,
                'densidad_construccion': float(densidad_construccion),
                'tipo_bien': datos_propiedad['desc_tipo_bien'],
                'regularizado': datos_propiedad['regularizado'],
                'superficie_edificada_m2': datos_propiedad['sup_edificada'],
                'superficie_terreno_m2': datos_propiedad['sup_terreno'],
                'latitud': datos_propiedad['latitud'],
                'longitud': datos_propiedad['longitud'],
                'nombre_entidad': datos_propiedad['nombre_entidad']
            }
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error en predict_encargo_endpoint: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/encargo_metrics', methods=['GET'])
def encargo_metrics_endpoint():
    try:
        # Cargar métricas desde archivo JSON
        with open('encargos_data/metricas_encargo.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data), 200
    except FileNotFoundError:
        return jsonify({'error': 'metricas_encargo.json no encontrado en encargos_data/'}), 404
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Error al parsear JSON: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f"Error en encargo_metrics_endpoint: {e}")
        return jsonify({'error': str(e)}), 500





if __name__ == '__main__':
    # Leer variables de entorno para Flask
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 8080))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host=host, port=port, debug=debug)
