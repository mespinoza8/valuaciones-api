FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gcc g++ libpq-dev \
      gdal-bin libgdal-dev && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m apiuser

WORKDIR /app


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py model.py utils.py data_metrics.py modelo_valoracion.pkl .env ./
COPY comunas.xlsx resultados_qa.xlsx ./
COPY data_preprocessed/ ./data_preprocessed/
COPY metrics.json ./
COPY Makefile ./



RUN chown -R apiuser:apiuser /app
USER apiuser

EXPOSE 8080
HEALTHCHECK CMD curl -f http://localhost:8080/metrics || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"] 