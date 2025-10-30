# Guía de Solución: Docker Build Timeout

## Problema

El build de Docker se queda atascado durante la instalación de `libgdal-dev` y otros paquetes del sistema.

## Soluciones Implementadas

### 1. Dockerfile Actualizado
- ✅ Agregado `shap_explainer.py` al COPY
- ✅ Agregado `curl` para healthcheck
- ✅ Optimizado orden de instalación

### 2. Requirements.txt Actualizado
- ✅ Agregado `shap==0.49.1`
- ✅ Agregado `numba>=0.54`
- ✅ Agregado `tqdm>=4.27.0`

### 3. .dockerignore Creado
- ✅ Evita copiar archivos innecesarios
- ✅ Reduce tamaño del contexto de build

## Pasos para Resolver el Timeout

### Opción 1: Limpiar Caché de Docker y Reintentar

```bash
# 1. Detener contenedores
make stop

# 2. Limpiar caché de Docker
docker system prune -a --volumes -f

# 3. Intentar build nuevamente con más tiempo
DOCKER_BUILDKIT=1 docker build --no-cache --progress=plain -t valuaciones-api:latest .
```

### Opción 2: Build con BuildKit (Recomendado)

```bash
# BuildKit es más eficiente y maneja mejor los timeouts
export DOCKER_BUILDKIT=1
make build
```

### Opción 3: Aumentar Timeout de Docker

Si estás en macOS con Docker Desktop:

1. Abrir Docker Desktop
2. Settings → Resources → Advanced
3. Aumentar Memory a 4GB+ y Swap a 2GB+
4. Apply & Restart

### Opción 4: Build Multietapa (Si el problema persiste)

Si aún hay problemas, podemos cambiar a un Dockerfile multietapa:

```dockerfile
# Etapa 1: Build dependencies
FROM python:3.10-slim as builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gcc g++ libpq-dev \
      gdal-bin libgdal-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Etapa 2: Runtime
FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libpq5 gdal-bin curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# ... resto del Dockerfile
```

## Verificar el Build

### Monitor del Progreso

```bash
# Ver progreso detallado
docker build --progress=plain -t valuaciones-api:latest . 2>&1 | tee build.log
```

### Si el Build Completa

```bash
# Verificar imagen
docker images | grep valuaciones-api

# Probar contenedor
make run

# Ver logs
make logs

# Probar endpoint
curl http://localhost:8080/metrics
```

## Prueba Rápida sin Docker

Si necesitas probar la implementación de SHAP sin esperar el build de Docker:

```bash
# Activar entorno virtual
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Iniciar API localmente
python app.py

# En otra terminal, probar
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitud": -33.4489,
    "longitud": -70.6693,
    "tipo": "casa",
    "superficie_util": 100,
    "superficie_total": 150,
    "dormitorios": 3,
    "banos": 2,
    "comuna": "santiago"
  }'
```

## Alternativa: Docker Compose con Cache

Si el problema persiste, podemos usar docker-compose con cache:

```yaml
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
      cache_from:
        - valuaciones-api:latest
    image: valuaciones-api:latest
    ports:
      - "8080:8080"
    environment:
      - FLASK_ENV=production
```

## Troubleshooting

### Error: "apt-get timeout"
```bash
# Usar mirrors más rápidos
docker build --build-arg DEBIAN_FRONTEND=noninteractive \
  --network=host -t valuaciones-api:latest .
```

### Error: "No space left on device"
```bash
# Limpiar espacio
docker system df
docker system prune -a --volumes
```

### Error: "killed" durante pip install
```bash
# Aumentar memoria de Docker a 4GB+
# Ver Settings → Resources en Docker Desktop
```

## Verificación Final

Una vez completado el build:

```bash
# 1. Verificar imagen
docker images valuaciones-api

# 2. Iniciar contenedor
make run

# 3. Verificar salud
docker ps -a | grep valuaciones-api

# 4. Ver logs de inicio
make logs

# 5. Probar SHAP
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

## Contacto

Si el problema persiste después de intentar estas soluciones, por favor proporciona:
1. Output completo del build: `docker build ... 2>&1 | tee build.log`
2. Versión de Docker: `docker --version`
3. Sistema operativo
4. Espacio disponible: `df -h` (Linux/Mac) o `docker system df`
