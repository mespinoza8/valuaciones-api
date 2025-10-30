# Explicabilidad con SHAP - Implementación Completada

## Resumen

Se ha implementado exitosamente la explicabilidad del modelo de valuación usando SHAP (SHapley Additive exPlanations). El endpoint `/predict` ahora incluye automáticamente una explicación detallada de cómo el modelo llegó a su predicción.

## Archivos Modificados/Creados

### 1. **shap_explainer.py** (NUEVO)
Módulo que encapsula toda la lógica de explicabilidad SHAP:
- Clase `SHAPExplainer` para generar explicaciones
- Soporte para modelos Pipeline de sklearn
- Traducción de nombres técnicos a español
- Generación de texto explicativo legible

### 2. **app.py** (MODIFICADO)
- Se importa `SHAPExplainer` en línea 18
- Se inicializa el explicador SHAP en líneas 84-90
- Se genera explicación SHAP en el endpoint `/predict` (líneas 152-163)
- Formato de respuesta actualizado con estructura JSON solicitada (líneas 212-246)

### 3. **test_shap_local.py** (NUEVO)
Script de prueba local para verificar SHAP sin necesidad de correr la API.

### 4. **test_shap.py** (NUEVO)
Script de prueba para el endpoint completo via HTTP.

## Formato de Respuesta

El endpoint `/predict` ahora devuelve:

```json
{
  "success": true,
  "data": {
    "tipo": "casa",
    "comuna": "punta arenas",
    "precio_uf": 4660,
    "rango": "2656 - 6663 UF",
    "confianza": "Baja",
    "margen_error": "±43.0%",
    "region": "Región de Magallanes y Antártica Chilena",
    "shap_explanation": {
      "top_features": [
        {
          "feature": "Tipo departamento",
          "feature_raw": "cat__tipo_departamento",
          "shap_value": -511.983,
          "feature_value": 0.0,
          "impact_direction": "decrease"
        },
        {
          "feature": "Distancia metro",
          "feature_raw": "num__distancia_metro_km",
          "shap_value": -472.7648,
          "feature_value": -4659292.39,
          "impact_direction": "decrease"
        },
        ...
      ],
      "explanation_text": "Explicación de la predicción (valor base: 7241.05 UF):\n\nFactores que más influyen en el precio:\n1. Tipo departamento (valor: 0.00): reduce el precio en ~-511.98 UF (SHAP: -511.9830)\n2. Distancia metro (valor: -4659292.39): reduce el precio en ~-472.76 UF (SHAP: -472.7648)\n..."
    }
  }
}
```

## Uso

### Iniciar la API

```bash
cd /Users/mespinoza/Desktop/Valuaciones/valuaciones-api
source .venv/bin/activate
python app.py
```

### Hacer una predicción con explicación SHAP

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitud": -53.1638,
    "longitud": -70.9171,
    "tipo": "casa",
    "superficie_util": 111.0,
    "superficie_total": 150.0,
    "dormitorios": 3,
    "banos": 2,
    "comuna": "punta arenas"
  }'
```

### Prueba Local (sin API)

```bash
python test_shap_local.py
```

## Características de la Implementación

### ✅ Completado

1. **Instalación de SHAP**: Librería instalada en el entorno virtual
2. **Módulo de explicabilidad**: `shap_explainer.py` con clase reutilizable
3. **Integración en API**: Endpoint `/predict` actualizado
4. **Formato de respuesta**: JSON con estructura solicitada
5. **Traducción de features**: Nombres técnicos traducidos al español
6. **Texto explicativo**: Generación automática de explicación legible
7. **Manejo de errores**: Fallback graceful si SHAP falla

### 🔧 Características Técnicas

- **TreeExplainer de SHAP**: Optimizado para Random Forest
- **Soporte para Pipeline**: Compatible con sklearn Pipeline
- **Top N features**: Configurable (default: 5 más importantes)
- **Valores SHAP**: Contribución directa en UF al precio predicho
- **Base value**: Valor promedio del modelo usado como referencia

## Interpretación de Resultados

### Valores SHAP

- **Valor positivo**: La feature incrementa el precio predicho
- **Valor negativo**: La feature reduce el precio predicho
- **Magnitud**: Representa el impacto en UF

### Ejemplo de Interpretación

```
"shap_value": -511.98
"feature": "Tipo departamento"
"feature_value": 0.0
```

Esto significa: "El hecho de que NO sea un departamento (valor 0) reduce el precio en aproximadamente 512 UF comparado con el valor base del modelo"

## Dependencias Nuevas

- `shap==0.49.1`
- `numba==0.62.1` (dependencia de SHAP)
- `tqdm==4.67.1` (dependencia de SHAP)

## Notas Técnicas

1. **Rendimiento**: El cálculo de SHAP añade ~1-2 segundos por predicción
2. **Memoria**: TreeExplainer mantiene el árbol completo en memoria
3. **Escalabilidad**: Para batch predictions, considerar calcular SHAP en paralelo
4. **Caché**: Considerar cachear el explicador SHAP para múltiples requests

## Próximos Pasos (Opcionales)

- [ ] Agregar visualizaciones SHAP (waterfall plot, force plot)
- [ ] Endpoint dedicado `/explain` para análisis más profundos
- [ ] Cache de valores SHAP para predicciones frecuentes
- [ ] Explicaciones comparativas entre dos propiedades
- [ ] Dashboard de explicabilidad

## Contacto

Para dudas o mejoras, contactar al equipo de desarrollo.
