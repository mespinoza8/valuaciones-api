# api.py
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



load_dotenv()


# Configuración
MODEL_PATH = os.getenv('MODEL_PATH', 'modelo_valoracion.pkl')
SHP_PATHS = {
    'ed_superior': os.getenv('ED_SUPERIOR_SHP', 'data_preprocessed/ed_superior.parquet').strip("'\""),
    'ed_escolar':  os.getenv('ED_ESCOLAR_SHP',  'data_preprocessed/ed_escolar.parquet').strip("'\""),
    'comisarias':  os.getenv('COMISARIAS_SHP',  'data_preprocessed/comisarias.parquet').strip("'\""),
    'salud':       os.getenv('SALUD_SHP',       'data_preprocessed/salud.parquet').strip("'\""),
    'metro':       os.getenv('METRO_SHP',       'data_preprocessed/metro.parquet').strip("'\""),
    'comunas':     os.getenv('COMUNAS_SHP',     'data_preprocessed/comunas.parquet').strip("'\"")
}


# --- Configuración de la Base de Datos MariaDB ---
db_user = os.getenv('DB_USER')
db_pass = os.getenv('DB_PASSWORD')
db_host = os.getenv('HOST')
db_port = os.getenv('DB_PORT', '3306')
db_name = os.getenv('DB_NAME', 'valuaciones')
# Añadimos charset utf8mb4 para compatibilidad de caracteres
DB_URI = (
    f"mysql+pymysql://"
    f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('HOST')}:{os.getenv('PORT','3306')}/"
    f"{os.getenv('DATABASE','ml_valoranet')}"
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
            Region:
              type: string
              example: Región Metropolitana de Santiago
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
        features = request.get_json(force=True)
        features['divisa'] = 'UF'
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


                # Guardar consulta y resultado en PostgreSQL
        record = {
            **features,
            'latitud': lat,
            'longitud': lon,
            'prediction_uf': float(prediction),
            'requested_at': datetime.utcnow()
        }
        insert_sql = text("""
            INSERT INTO model_predictions (
                divisa, tipo, superficie_util, superficie_total,
                antiguedad, dormitorios, banos,
                comuna, region, latitud, longitud,
                distancia_ed_superior_km, distancia_ed_escolar_km,
                distancia_comisaria_km, distancia_est_salud_km, distancia_metro_km,
                prediction_uf, requested_at
            ) VALUES (
                :divisa, :tipo, :superficie_util, :superficie_total,
                :antiguedad, :dormitorios, :banos,
                :Comuna, :Region, :latitud, :longitud,
                :distancia_ed_superior_km, :distancia_ed_escolar_km,
                :distancia_comisaria_km, :distancia_est_salud_km, :distancia_metro_km,
                :prediction_uf, :requested_at
            )
        """)
        with engine.begin() as conn:
            conn.execute(insert_sql, record)

        return jsonify({'prediction_uf': float(prediction)})

    except Exception as e:
        app.logger.error(f"Error in predict_endpoint: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Leer variables de entorno para Flask
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 8000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host=host, port=port, debug=debug)
