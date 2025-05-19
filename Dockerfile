FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc g++ libpq-dev \
    gdal-bin libgdal-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements y instalarlas
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código, el modelo y el .env
COPY app.py model.py utils.py modelo_valoracion.pkl .env ./

# Copiar los parquet preprocesados
COPY data_preprocessed/ ./data_preprocessed/

# Exponer el puerto de la API
EXPOSE 8000

# Arrancar con Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]