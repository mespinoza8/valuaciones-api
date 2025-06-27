import os
from dotenv import load_dotenv
import pandas as pd
import geopandas as gpd
from flask import Flask, request, jsonify
from utils import create_point_gdf, calculate_nearest_distances, calculate_nearest_distances_metro
import joblib
from sqlalchemy import create_engine, text
from datetime import datetime
from flasgger import Swagger
import subprocess
import jwt
import json
from data_metrics import metricas_comuna

load_dotenv()


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
COMUNA_REGION_MAP = dict(zip(map_df['Comuna'], map_df['Region']))



# --- Configuración de la Base de Datos MariaDB ---
db_user = os.getenv('DB_USER')
db_pass = os.getenv('DB_PASSWORD')
db_host = os.getenv('HOST')
db_port = os.getenv('DB_PORT', '3306')
db_name = os.getenv('DB_NAME', 'ml_valoranet')
# Añadimos charset utf8mb4 para compatibilidad de caracteres
DB_URI = (
    f"mysql+pymysql://{db_user}:{db_pass}"
    f"@{db_host}:{db_port}/{db_name}"
)
engine = create_engine(DB_URI)



# Cargar el modelo serializado
model = joblib.load(MODEL_PATH)

# Cargar shapefiles en memoria
ed_sup = gpd.read_parquet(SHP_PATHS['ed_superior'])
ed_esc = gpd.read_parquet(SHP_PATHS['ed_escolar'])
comi   = gpd.read_parquet(SHP_PATHS['comisarias'])
salud  = gpd.read_parquet(SHP_PATHS['salud'])
metro  = gpd.read_parquet(SHP_PATHS['metro'])

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
# Documentación de la API
#http://localhost:8000/apidocs/
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Estima el valor de una propiedad en UF.
    ---
    tags:
      - Valuaciones
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            tipo:
              type: string
              example: departamento
            superficie_util:
              type: number
              example: 61
            superficie_total:
              type: number
              example: 64
            antiguedad:
              type: number
              example: 30
            dormitorios:
              type: integer
              example: 2
            banos:
              type: integer
              example: 1
            Comuna:
              type: string
              example: Cerrillos
            latitud:
              type: number
              example: -33.514
            longitud:
              type: number
              example: -70.707
    responses:
      200:
        description: Valor estimado en UF
        schema:
          type: object
          properties:
            prediction_uf:
              type: number
              example: 4523.17
      400:
        description: JSON malformado o error de solicitud
    """



    try:
      data = request.get_json(force=True)
      comuna = data.get('Comuna')
      if not comuna:
          return jsonify({'error': "Debes enviar la 'Comuna'"}), 400

      # Autocompletar región
      region = COMUNA_REGION_MAP.get(comuna)
      if not region:
          return jsonify({'error': f"Comuna '{comuna}' no está permitida"}), 400

      # Armar el payload con región añadida
      features = {**data, 'Region': region, 'divisa': 'UF'}

      lat = features['latitud']
      lon = features['longitud']
      punto = create_point_gdf(lat, lon)

        # Calcular distancias
      features['distancia_ed_superior_km'] = calculate_nearest_distances(punto, ed_sup)[0]
      features['distancia_ed_escolar_km']  = calculate_nearest_distances(punto, ed_esc)[0]
      features['distancia_comisaria_km']   = calculate_nearest_distances(punto, comi)[0]
      features['distancia_est_salud_km']   = calculate_nearest_distances(punto, salud)[0]
      features['distancia_metro_km']       = calculate_nearest_distances_metro(punto, metro)[0]

      # Crear DataFrame y predecir
      df_new = pd.DataFrame([features])
      prediction = model.predict(df_new)[0]

      # Calcular métricas por comuna


      metricas_comuna_df = metricas_comuna()
      comuna_metrics = metricas_comuna_df.query("Comuna == @comuna")
      avg_price_uf=comuna_metrics.query("Comuna == @comuna")['avg_price_uf'].values[0]
      superficie_util_promedio=comuna_metrics.query("Comuna == @comuna")['superficie'].values[0]
      nro_propiedades=comuna_metrics.query("Comuna == @comuna")['n_properties'].values[0]




              # Guardar consulta y resultado en BD
      record = {
          **features,
          'antiguedad':10,
          'latitud': lat,
          'longitud': lon,
          'prediction_uf': float(prediction),
          'avg_price_uf':    avg_price_uf,
          'avg_price_uf_m2': superficie_util_promedio,
          'n_properties':  nro_propiedades,
          'requested_at': datetime.now()
      }
      insert_sql = text("""
          INSERT INTO model_predictions (
              divisa, tipo, superficie_util, superficie_total,
              antiguedad, dormitorios, banos,
              comuna, region, latitud, longitud,
              distancia_ed_superior_km, distancia_ed_escolar_km,
              distancia_comisaria_km, distancia_est_salud_km, distancia_metro_km,
              prediction_uf,avg_price_uf,avg_price_uf_m2, n_properties,requested_at
          ) VALUES (
              :divisa, :tipo, :superficie_util, :superficie_total,
              :antiguedad, :dormitorios, :banos,
              :Comuna, :Region, :latitud, :longitud,
              :distancia_ed_superior_km, :distancia_ed_escolar_km,
              :distancia_comisaria_km, :distancia_est_salud_km, :distancia_metro_km,
              :prediction_uf, :avg_price_uf, :avg_price_uf_m2, :n_properties, :requested_at
          )
      """)
      with engine.begin() as conn:
          conn.execute(insert_sql, record)

      return jsonify({'prediction_uf': float(prediction),
                      'valor_promedio_propiedades_comuna':float(avg_price_uf),
                      'superficie_util_promedio_comuna' :float(superficie_util_promedio),
                      'cantidad_propiedades_comuna': int(nro_propiedades)})

    except Exception as e:
        app.logger.error(f"Error in predict_endpoint: {e}")
        return jsonify({'error': str(e)}), 400
    


# --- Nuevo endpoint para reentrenar el modelo ---
@app.route('/retrain', methods=['POST'])
def retrain_endpoint():
    """
    Reentrena el modelo y actualiza el archivo serializado.
    ---
    tags:
      - Valuaciones
    parameters:
      - in: header
        name: Authorization
        required: true
        description: Token JWT en formato `Bearer <token>`
        schema:
          type: string
    responses:
      200:
        description: Modelo reentrenado con éxito
        schema:
          type: object
          properties:
            status:
              type: string
              example: success
            message:
              type: string
              example: Model retrained
            token:
              type: string
              example: eyJ0eXAiOiJKV1QiLCJh...
      401:
        description: Falta o token inválido
        schema:
          type: object
          properties:
            error:
              type: string
              example: Missing or invalid Authorization header
      500:
        description: Error al reentrenar el modelo
        schema:
          type: object
          properties:
            status:
              type: string
              example: error
            message:
              type: string
              example: Traceback (most recent call last)...
    """

    # Autenticación
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return jsonify(error='Missing or invalid Authorization header'), 401
    token = auth.split(' ', 1)[1]
    try:
        jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    except jwt.InvalidTokenError:
        return jsonify(error='Invalid token'), 401

    # Lanzar entrenamiento (nota: esto bloqueará la petición)
    proc = subprocess.run(['python3', 'train_model.py'], capture_output=True, text=True)
    if proc.returncode != 0:
        return jsonify(status='error', message=proc.stderr), 500

    # Recargar modelo
    global model
    model = joblib.load(MODEL_PATH)

    # Devolver token nuevo opcionalmente
    new_token = jwt.encode({'retrained_at': datetime.now().isoformat()}, SECRET_KEY, algorithm='HS256')
    return jsonify(status='success', message='Model retrained', token=new_token), 200

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




if __name__ == '__main__':
    # Leer variables de entorno para Flask
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 8080))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host=host, port=port, debug=debug)
