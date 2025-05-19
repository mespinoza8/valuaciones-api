# Valuaciones API

Esta API permite estimar el valor de una propiedad en UF (Unidad de Fomento) a partir de sus características y su ubicación. Cada consulta se almacena en una base de datos MariaDB.

---

## Tabla de Contenidos

* [Características](#características)
* [Requisitos](#requisitos)
* [Instalación](#instalación)
* [Configuración de variables de entorno](#configuración-de-variables-de-entorno)
* [Ejecución local](#ejecución-local)
* [Docker](#docker)
* [Endpoints](#endpoints)

  * [`POST /predict`](#post-predict)
* [Ejemplos de uso](#ejemplos-de-uso)
* [Esquema de la tabla `predictions`](#esquema-de-la-tabla-predictions)
* [Swagger UI](#swagger-ui)

---

## Características

* Recepción de las características de la propiedad (tipo, superficies, antigüedad, dormitorios, baños, comuna, región, coordenadas).
* Cálculo automático de distancias a colegios, comisarías, salud y metro a partir de datos geoespaciales preprocesados en Parquet.
* Predicción del valor en UF usando diferentes modelos (LightGBM,Catboost, RandomForest,Multi-layer Perceptron regressor).
* Almacenamiento de cada consulta y resultado en MariaDB.
* Documentación interactiva con Swagger UI.

---

## Requisitos

* Python 3.10+
* MariaDB/MySQL accesible (directo o vía VPN)
* Docker y Docker Compose (opcional)

Dependencias Python listadas en `requirements.txt`:

```
Flask
pandas
numpy
geoPandas
shapely
scikit-learn
lightgbm
catboost
sqlalchemy
pymysql
python-dotenv
pyarrow
joblib
gunicorn
```

---

## Instalación

1. Clona el repositorio:

   ```bash
   git clone https://github.com/tuusuario/valuaciones-api.git
   cd valuaciones-api
   ```
2. Crea un entorno virtual e instálalo:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Configuración de variables de entorno

Crea un archivo `.env` en la raíz con al menos:

```dotenv
# Credenciales MariaDB
DB_USER=usuario_db
DB_PASSWORD=contraseña_db
HOST=XXX.XXX.X.X      # IP o hostname de MariaDB
DB_PORT=3306
DB_NAME=base_de_datos

# Modelo y datos preprocesados (archivos Parquet)
MODEL_PATH=modelo_valoracion.pkl
ED_SUPERIOR_SHP=data_preprocessed/ed_superior.parquet
ED_ESCOLAR_SHP=data_preprocessed/ed_escolar.parquet
COMISARIAS_SHP=data_preprocessed/comisarias.parquet
SALUD_SHP=data_preprocessed/salud.parquet
METRO_SHP=data_preprocessed/metro.parquet
COMUNAS_SHP=data_preprocessed/comunas.parquet

# Flask
FLASK_HOST=0.0.0.0
FLASK_PORT=8000
FLASK_DEBUG=true


---

## Ejecución local

Con la VPN activa (si aplica) y el entorno virtual:

```bash
source .venv/bin/activate
python app.py
```

La API quedará disponible en `http://localhost:8000`.

---

## Docker

### Dockerfile

El proyecto incluye un `Dockerfile` que instala dependencias, copia el código, el modelo y los Parquet, y arranca Gunicorn:

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gcc g++ libpq-dev gdal-bin libgdal-dev \
    openconnect iproute2 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install pyarrow gunicorn
COPY app.py model.py utils.py modelo_valoracion.pkl .env ./
COPY data_preprocessed/ ./data_preprocessed/
# COPY vpn/ /vpn/  # si encapsulas VPN

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
```

### Docker Compose

Archivo `docker-compose.yml`:

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped
```

### Construir y ejecutar

```bash
docker-compose build
docker-compose up -d
```

---

## Endpoints

### POST `/predict`

Estimación del valor en UF.

#### Request Body (JSON)

| Campo              | Tipo      | Descripción                         |
| ------------------ | --------- | ----------------------------------- |
| `tipo`             | `string`  | Tipo de propiedad (`departamento`). |
| `superficie_util`  | `number`  | m² superficie útil.                 |
| `superficie_total` | `number`  | m² superficie total.                |
| `antiguedad`       | `number`  | Años de antigüedad.                 |
| `dormitorios`      | `integer` | Nº de dormitorios.                  |
| `banos`            | `integer` | Nº de baños.                        |
| `Comuna`           | `string`  | Nombre de comuna.                   |
| `Region`           | `string`  | Nombre de región.                   |
| `latitud`          | `number`  | Latitud decimal.                    |
| `longitud`         | `number`  | Longitud decimal.                   |

#### Response Body (JSON)

| Campo           | Tipo     | Descripción           |
| --------------- | -------- | --------------------- |
| `prediction_uf` | `number` | Valor estimado en UF. |

**Ejemplo**:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
        "divisa":"UF",
        "tipo":"departamento",
        "superficie_util":61,
        "superficie_total":64,
        "antiguedad":30,
        "dormitorios":2,
        "banos":1,
        "Comuna":"Cerrillos",
        "Region":"Región Metropolitana de Santiago",
        "latitud":-33.514,
        "longitud":-70.707
      }'
```

---

## Ejemplos en Python

```python
import requests

url = "http://localhost:8000/predict"
payload = { ... }  # como arriba
resp = requests.post(url, json=payload)
print(resp.json())
```

---

## Esquema de la tabla `model_predictions`

```sql
CREATE TABLE IF NOT EXISTS predictions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  divisa            VARCHAR(10),
  tipo              VARCHAR(50),
  superficie_util   DECIMAL(12,2),
  superficie_total  DECIMAL(12,2),
  antiguedad        DECIMAL(12,2),
  dormitorios       INT,
  banos             INT,
  comuna            VARCHAR(100),
  region            VARCHAR(100),
  latitud           DOUBLE,
  longitud          DOUBLE,
  distancia_ed_superior_km  DOUBLE,
  distancia_ed_escolar_km   DOUBLE,
  distancia_comisaria_km    DOUBLE,
  distancia_est_salud_km    DOUBLE,
  distancia_metro_km        DOUBLE,
  prediction_uf     DECIMAL(12,4),
  requested_at      DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

---

## Swagger UI

La documentación interactiva está disponible en:

```
http://<host>:<port>/apidocs/
```

Sustituir `<host>` y `<port>` según el despliegue.
