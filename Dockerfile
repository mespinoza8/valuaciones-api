FROM python:3.10-slim

# Instalar dependencias del sistema con timeout aumentado
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gcc g++ libpq-dev curl \
      gdal-bin libgdal-dev && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m apiuser

WORKDIR /app

# Copiar requirements y instalar dependencias Python
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivos de la aplicación
COPY app.py model.py utils.py data_metrics.py modelo_valoracion.joblib .env ./
COPY shap_explainer.py ./
COPY comunas.xlsx resultados_qa.xlsx ./
COPY data_preprocessed/ ./data_preprocessed/
COPY metrics.json ./
COPY Makefile ./

# Archivos del modelo v2 para encargos
COPY encargos_data/ ./encargos_data/
COPY encargos_model.py ./

RUN chown -R apiuser:apiuser /app
USER apiuser

EXPOSE 8080
HEALTHCHECK CMD curl -f http://localhost:8080/metrics || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"] 