FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gcc g++ libpq-dev \
    gdal-bin libgdal-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py model.py utils.py modelo_valoracion.pkl .env ./

COPY data_preprocessed/ ./data_preprocessed/

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]