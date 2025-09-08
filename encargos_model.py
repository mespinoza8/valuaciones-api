
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
import warnings
import joblib
import os
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sqlalchemy import text,create_engine
from datetime import datetime

warnings.filterwarnings('ignore')

load_dotenv()



def cargar_datos():
    """Cargar y preparar los datos"""
    print("=== CARGANDO DATOS ===")

    query=text("""WITH X AS (SELECT DISTINCT
    fecha_estado_encargo,
    a.id_encargo,
    b.id_dato_tecnico_encargo,
    a.nombre_entidad,
    a.desc_finalidad,
    case when a.desc_tipo_bien ='DEPARTAMENTO' then 'DEPARTAMENTO' else 'CASA' end as desc_tipo_bien,
  a.desc_tipo_bien as desc_tipo_bien_original,
  concat(a.calle_bien,' ',a.numero_bien,' ',a.casa_depto_bien) as direccion,
  a.comuna_bien,
  a.ROL01,
    a.latitud,
    a.longitud,
    a.valor_comercial_encargo_supervisado_uf,
    b.ano_construccion,
    b.material,
    b.regularizado,
    b.sup_edificada,
    b.sup_terreno,
    b.antiguedad_construccion,
    b.destino_sii,
    b.uso_actual

FROM ml_valoranet.encargo a
   JOIN ml_valoranet.dato_tecnico_encargo b ON a.id_encargo = b.id_encargo

  WHERE YEAR(FECHA_ESTADO_ENCARGO) != '0'
    AND a.desc_finalidad != 'TASACION DE PRUEBA'
    AND a.VALOR_COMERCIAL_ENCARGO_SUPERVISADO_UF>0
    AND a.DESC_TIPO_BIEN in (
  'VIVIENDA_UNIFAMILIAR',
  'DEPARTAMENTO',
  'CASA')
  AND a.valor_comercial_encargo_supervisado_uf>0
  AND tipo_construccion IN ('CA','CASA','DEPARTAMENTO','DP')

  and length(b.ano_construccion)=4
  and uso_actual in ('Habitacion','HABITACIONAL','Habitación')
)  select * from x""")

    db_user = os.getenv('DB_USER')
    db_pass = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT', '3306')
    db_name = os.getenv('DB_NAME', 'ml_valoranet')
    DB_URI = (
        f"mysql+pymysql://{db_user}:{db_pass}"
        f"@{db_host}:{db_port}/{db_name}"
    )
    
    engine = create_engine(DB_URI)


    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    df_modelo=df.copy()

    df_modelo=df_modelo[['desc_tipo_bien','latitud','longitud','ano_construccion','material','regularizado',
'sup_edificada','sup_terreno','antiguedad_construccion',
'destino_sii','uso_actual','valor_comercial_encargo_supervisado_uf','nombre_entidad']]
    
    df_modelo['valor_corregido']=df_modelo['valor_comercial_encargo_supervisado_uf'].str.replace('.', '', regex=False)\
    .str.replace(',', '.', regex=False).astype(float)

    def eliminar_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        for col in cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
        return df

    df_modelo=eliminar_outliers_iqr(df_modelo, ['valor_corregido'])
    df_modelo=df_modelo[df_modelo['valor_corregido']>=800]
    df_modelo=df_modelo[df_modelo['ano_construccion'].astype(str).str.strip().str.match(r'^\d{4}$')]
    df_modelo['ano_construccion'] = pd.to_numeric(df_modelo['ano_construccion'], errors='coerce')
    df_modelo=df_modelo[df_modelo['ano_construccion']<=datetime.now().year]

    df_modelo['regularizado']=df_modelo['regularizado'].apply(lambda x: x if x in ['SI','NO','PARCIAL'] else 'SIN INFORMACION')

    df_modelo['latitud_corregida'] = pd.to_numeric(df_modelo['latitud'], errors='coerce')
    df_modelo['longitud_corregida'] = pd.to_numeric(df_modelo['longitud'], errors='coerce')

    mask_coords = (
    (df_modelo['latitud_corregida'] >= -56.5) & 
    (df_modelo['latitud_corregida'] <= -17.5) & 
    (df_modelo['longitud_corregida'] >= -80) & 
    (df_modelo['longitud_corregida'] <= -66))

    df_modelo=df_modelo[mask_coords]
    df_modelo['sup_edificada']=pd.to_numeric(df_modelo['sup_edificada'],errors='coerce')
    df_modelo['sup_terreno']=pd.to_numeric(df_modelo['sup_edificada'],errors='coerce')
    df_modelo['ano_construccion'] = pd.to_numeric(df_modelo['ano_construccion'], errors='coerce')

    df_modelo=df_modelo.query("sup_edificada.notna()")
    cols=['desc_tipo_bien','latitud_corregida','longitud_corregida','ano_construccion','regularizado',
'sup_edificada','sup_terreno','valor_corregido','nombre_entidad']

    df_modelo=df_modelo[cols]

    
    return df_modelo


def eliminar_outliers_ano_construccion(df):
    """Eliminar outliers en años de construcción"""
    print("\n=== ELIMINACIÓN DE OUTLIERS EN AÑOS DE CONSTRUCCIÓN ===")
    
    # Estadísticas antes de la limpieza
    print(f"Registros antes de limpiar outliers: {len(df)}")
    print(f"Rango de años de construcción: {df['ano_construccion'].min()} - {df['ano_construccion'].max()}")
    
    # Calcular estadísticas para detectar outliers
    Q1 = df['ano_construccion'].quantile(0.25)
    Q3 = df['ano_construccion'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Definir límites para outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    current_year = 2024
    business_lower = 1900
    business_upper = current_year + 2
    
    # Usar el límite más restrictivo
    final_lower = max(lower_bound, business_lower)
    final_upper = min(upper_bound, business_upper)
    
    print(f"Límites finales: {final_lower:.0f} - {final_upper:.0f}")
    
    # Identificar outliers
    outliers_mask = (df['ano_construccion'] < final_lower) | (df['ano_construccion'] > final_upper)
    outliers_count = outliers_mask.sum()
    
    print(f"Outliers detectados: {outliers_count} ({outliers_count/len(df)*100:.2f}%)")
    
    # Eliminar outliers
    df_clean = df[~outliers_mask].copy()
    
    print(f"Registros después de limpiar outliers: {len(df_clean)}")
    
    return df_clean

def realizar_eda(df):
    """Realizar análisis exploratorio de datos"""
    print("\n=== ANÁLISIS EXPLORATORIO DE DATOS ===")
    
    # Información general
    print(f"Forma del dataset: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    
    print(f"\nVariable objetivo: valor_corregido")
    print(f"Rango de valores: {df['valor_corregido'].min():.2f} - {df['valor_corregido'].max():.2f} UF")
    print(f"Media: {df['valor_corregido'].mean():.2f} UF")
    print(f"Mediana: {df['valor_corregido'].median():.2f} UF")
    
    # Verificar correlaciones con la variable objetivo (solo variables numéricas)
    print(f"\nCorrelaciones con valor_corregido:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['valor_corregido'].sort_values(ascending=False)
    print(correlations)
    
    # Verificar si existen las variables que queremos excluir
    excluded_vars = ['uf_m2_edificada', 'uf_m2_terreno']
    for var in excluded_vars:
        if var in df.columns:
            print(f" Variable {var} encontrada - será excluida del modelo")
            correlation = df[var].corr(df['valor_corregido'])
            print(f"   Correlación con valor_corregido: {correlation:.4f}")
    
    return correlations

def feature_engineering(df):
    """Realizar feature engineering sin data leakage"""
    print("\n=== FEATURE ENGINEERING (SIN DATA LEAKAGE) ===")
    
    
    df_fe = df.copy()
    
    # 1. Features de densidad (sin usar valor_corregido)
    df_fe['densidad_construccion'] = df_fe['sup_edificada'] / df_fe['sup_terreno'].replace(0, np.nan)
    df_fe['densidad_construccion'] = df_fe['densidad_construccion'].fillna(1)
    
    # 2. Features temporales
    df_fe['antiguedad'] = 2024 - df_fe['ano_construccion']
    
    # 3. Bins de antigüedad
    df_fe['antiguedad_bin'] = pd.cut(df_fe['antiguedad'], 
                                    bins=[0, 10, 20, 30, 50, 100], 
                                    labels=['0-10', '11-20', '21-30', '31-50', '50+'])
    
    # 4. Bins de superficie
    df_fe['sup_edificada_bin'] = pd.cut(df_fe['sup_edificada'], 
                                       bins=[0, 50, 80, 120, 200, 1000], 
                                       labels=['0-50', '51-80', '81-120', '121-200', '200+'])
    
    # 5. Clustering geográfico (sin usar valor_corregido)
    print("Creando clusters geográficos...")
    coords = df_fe[['latitud_corregida', 'longitud_corregida']].dropna()
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df_fe['zona_geografica'] = kmeans.fit_predict(coords)
    
    # Guardar el modelo KMeans para uso futuro
    joblib.dump(kmeans, 'encargos_data/kmeans_zona_geografica_v2.pkl')
    print("Modelo KMeans guardado como 'encargos_data/kmeans_zona_geografica_v2.pkl'")
    
    # 6. Distancia al centro
    centro_lat, centro_lon = -33.4489, -70.6693
    df_fe['distancia_centro'] = np.sqrt(
        (df_fe['latitud_corregida'] - centro_lat)**2 + 
        (df_fe['longitud_corregida'] - centro_lon)**2
    ) * 111
    
    # 7. Features de zona (sin usar valor_corregido)
    densidad_por_zona = df_fe.groupby('zona_geografica').size() / df_fe.groupby('zona_geografica').size().sum()
    df_fe['densidad_zona'] = df_fe['zona_geografica'].map(densidad_por_zona)
    
    # Guardar el mapeo de densidad por zona para uso futuro
    joblib.dump(densidad_por_zona, 'encargos_data/densidad_por_zona_v2.pkl')
    print("Mapeo de densidad por zona guardado como 'encargos_data/densidad_por_zona_v2.pkl'")
    
    # 8. Codificación de variables categóricas
    categorical_cols = ['desc_tipo_bien', 'regularizado', 'antiguedad_bin', 'sup_edificada_bin','nombre_entidad']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df_fe.columns:
            le = LabelEncoder()
            df_fe[f'{col}_encoded'] = le.fit_transform(df_fe[col].astype(str))
            label_encoders[col] = le
    
    print("Feature engineering completado (sin data leakage)")
    return df_fe, label_encoders

def preparar_datos(df_fe):
    """Preparar datos para el modelo sin data leakage"""
    print("\n=== PREPARACIÓN DE DATOS (SIN DATA LEAKAGE) ===")
    
    # Seleccionar features (excluyendo uf_m2_edificada, uf_m2_terreno y valor_corregido)
    feature_cols = [
        'latitud_corregida', 'longitud_corregida', 'ano_construccion',
        'sup_edificada', 'sup_terreno',
        'densidad_construccion', 'antiguedad',
        'zona_geografica', 'distancia_centro', 'densidad_zona',
        'desc_tipo_bien_encoded', 'regularizado_encoded', 
        'antiguedad_bin_encoded', 'sup_edificada_bin_encoded','nombre_entidad_encoded'
    ]
    
    # Verificar que no estén las variables excluidas
    excluded_vars = ['uf_m2_edificada', 'uf_m2_terreno', 'valor_corregido']
    for var in excluded_vars:
        if var in feature_cols:
            feature_cols.remove(var)
            print(f"  Variable {var} removida de features")
    
    available_features = [col for col in feature_cols if col in df_fe.columns]
    print(f"Features disponibles: {len(available_features)}")
    print(f"Features: {available_features}")
    
    # Crear dataset final
    X = df_fe[available_features].dropna()
    y = df_fe.loc[X.index, 'valor_corregido']
    
    # Limpiar valores infinitos
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    y = y.loc[X.index]
    
    print(f"Dataset final después de limpieza: {X.shape}")
    
    # División train/test con estratificación por zona geográfica
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=df_fe.loc[X.index, 'zona_geografica']
        )
    except:
        # Si no se puede estratificar, usar división normal
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, available_features

def analizar_overfitting(model, X_train, y_train, X_test, y_test, model_name):
    """Analizar overfitting del modelo"""
    print(f"\n=== ANÁLISIS DE OVERFITTING: {model_name} ===")
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Métricas
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Gap de overfitting
    r2_gap = train_r2 - test_r2
    rmse_gap = test_rmse - train_rmse
    mae_gap = test_mae - train_mae
    
    print(f"\nGap de Overfitting:")
    print(f"R² gap: {r2_gap:.4f}")
    print(f"RMSE gap: {rmse_gap:.4f}")
    print(f"MAE gap: {mae_gap:.4f}")
    
    # Evaluación de overfitting
    if r2_gap > 0.1:
        print("  ADVERTENCIA: Posible overfitting detectado (R² gap > 0.1)")
    elif r2_gap > 0.05:
        print(" ADVERTENCIA: Ligero overfitting detectado (R² gap > 0.05)")
    else:
        print("Modelo bien generalizado")
    
    # Learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2'
    )
    
    
    return {
        'model_name': model_name,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'r2_gap': r2_gap,
        'rmse_gap': rmse_gap,
        'mae_gap': mae_gap
    }

def analizar_data_leakage(df_fe, available_features):
    """Analizar posibles fuentes de data leakage"""
    print("\n=== ANÁLISIS DE DATA LEAKAGE ===")
    
    # 1. Verificar correlaciones con la variable objetivo
    print("1. Correlaciones con valor_corregido:")
    correlations = df_fe[available_features + ['valor_corregido']].corr()['valor_corregido'].sort_values(ascending=False)
    print(correlations)
    
    # 2. Verificar si hay features que contienen información del futuro
    print("\n2. Verificación de features temporales:")
    temporal_features = ['ano_construccion', 'antiguedad']
    for feature in temporal_features:
        if feature in available_features:
            correlation = df_fe[feature].corr(df_fe['valor_corregido'])
            print(f"   {feature}: correlación = {correlation:.4f}")
    
    # 3. Verificar features geográficas
    print("\n3. Verificación de features geográficas:")
    geo_features = ['latitud_corregida', 'longitud_corregida', 'distancia_centro', 'zona_geografica']
    for feature in geo_features:
        if feature in available_features:
            correlation = df_fe[feature].corr(df_fe['valor_corregido'])
            print(f"   {feature}: correlación = {correlation:.4f}")
    
    # 4. Verificar si hay features derivadas del valor
    print("\n4. Verificación de features derivadas:")
    derived_features = ['densidad_construccion', 'densidad_zona']
    for feature in derived_features:
        if feature in available_features:
            correlation = df_fe[feature].corr(df_fe['valor_corregido'])
            print(f"   {feature}: correlación = {correlation:.4f}")
    
    # 5. Verificar variables categóricas
    print("\n5. Verificación de variables categóricas:")
    categorical_features = ['desc_tipo_bien', 'regularizado', 'antiguedad_bin', 'sup_edificada_bin','nombre_entidad']
    for feature in categorical_features:
        if feature in available_features:
            # Calcular correlación usando la versión codificada
            encoded_feature = f'{feature}_encoded'
            if encoded_feature in available_features:
                correlation = df_fe[encoded_feature].corr(df_fe['valor_corregido'])
                print(f"   {feature}: correlación = {correlation:.4f}")
    
    # 6. Verificar que no estén las variables excluidas
    print("\n6. Verificación de variables excluidas:")
    excluded_vars = ['uf_m2_edificada', 'uf_m2_terreno', 'valor_corregido']
    for var in excluded_vars:
        if var in available_features:
            print(f"     ADVERTENCIA: {var} está en las features (debería estar excluida)")
        else:
            print(f"    {var} correctamente excluida")
    
    return correlations

def entrenar_modelos_con_analisis_overfitting(X_train, y_train, X_test, y_test):
    """Entrenar modelos con análisis de overfitting"""
    print("\n=== ENTRENAMIENTO DE MODELOS CON ANÁLISIS DE OVERFITTING ===")
    
    modelos = {}
    resultados_overfitting = []
    
    # 1. Random Forest
    print("\n--- Entrenando Random Forest ---")
    rf_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42, 
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    modelos['Random Forest'] = rf_model
    
    rf_analysis = analizar_overfitting(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
    resultados_overfitting.append(rf_analysis)
    
    # 2. XGBoost
    print("\n--- Entrenando XGBoost ---")
    xgb_model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42, 
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    modelos['XGBoost'] = xgb_model
    
    xgb_analysis = analizar_overfitting(xgb_model, X_train, y_train, X_test, y_test, "XGBoost")
    resultados_overfitting.append(xgb_analysis)
    
    # 3. Decision Tree
    print("\n--- Entrenando Decision Tree ---")
    dt_model = DecisionTreeRegressor(
        max_depth=6,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    modelos['Decision Tree'] = dt_model
    
    dt_analysis = analizar_overfitting(dt_model, X_train, y_train, X_test, y_test, "Decision Tree")
    resultados_overfitting.append(dt_analysis)
    
    # Resumen de resultados
    print("\n=== RESUMEN DE ANÁLISIS DE OVERFITTING ===")
    df_overfitting = pd.DataFrame(resultados_overfitting)
    print(df_overfitting[['model_name', 'train_r2', 'test_r2', 'r2_gap', 'train_rmse', 'test_rmse', 'rmse_gap']].round(4))
    
    return modelos, df_overfitting

def optimizar_mejor_modelo_sin_overfitting(df_overfitting, modelos, X_train, y_train, X_test, y_test):
    """Optimizar el mejor modelo evitando overfitting"""
    print("\n=== OPTIMIZACIÓN DEL MEJOR MODELO (SIN OVERFITTING) ===")
    
    # Seleccionar el modelo con menor gap de overfitting
    best_idx = df_overfitting['r2_gap'].idxmin()
    best_model_name = df_overfitting.loc[best_idx, 'model_name']
    best_gap = df_overfitting.loc[best_idx, 'r2_gap']
    
    print(f"Mejor modelo: {best_model_name} (gap de overfitting: {best_gap:.4f})")
    
    # Configurar cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    if best_model_name == "Random Forest":
        param_grid = {
            'n_estimators': [30, 50, 100],
            'max_depth': [6, 8, 10],
            'min_samples_split': [10, 15, 20],
            'min_samples_leaf': [4, 6, 8],
            'max_features': ['sqrt', 'log2']
        }
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    elif best_model_name == "XGBoost":
        param_grid = {
            'n_estimators': [30, 50, 100],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0.01, 0.1, 0.5],
            'reg_lambda': [0.5, 1.0, 2.0]
        }
        base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    else:
        param_grid = {
            'max_depth': [4, 6, 8],
            'min_samples_split': [15, 20, 25],
            'min_samples_leaf': [8, 10, 15],
            'max_features': ['sqrt', 'log2']
        }
        base_model = DecisionTreeRegressor(random_state=42)
    
    # Grid Search con Cross-Validation
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=cv, 
        scoring='r2', 
        n_jobs=-1, 
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score CV: {grid_search.best_score_:.4f}")
    
    # Verificar overfitting en el modelo optimizado
    best_model = grid_search.best_estimator_
    optimized_analysis = analizar_overfitting(best_model, X_train, y_train, X_test, y_test, f"{best_model_name} (Optimizado)")
    
    return best_model, optimized_analysis

def analizar_importancia_features(best_model, available_features):
    """Analizar importancia de features"""
    print("\n=== ANÁLISIS DE IMPORTANCIA DE FEATURES ===")
    
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
 
        
        print("\nTop 10 features más importantes:")
        print(feature_importance.head(10))
        
        return feature_importance
    
    return None

def guardar_importancia_variables_json(feature_importance, filename="encargos_data/importancia_variables_encargo.json"):
    """Guardar la importancia de variables en un archivo JSON"""
    import json
    # Convertir a lista de diccionarios
    importancia_list = feature_importance.to_dict(orient="records")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(importancia_list, f, indent=2, ensure_ascii=False)
    print(f"Importancia de variables guardada en {filename}")

def analizar_errores(y_test, y_pred):
    """Analizar errores del modelo"""
    print("\n=== ANÁLISIS DE ERRORES ===")
    
    errors = y_test - y_pred
    error_percentage = (errors / y_test) * 100
    
    print(f"\nEstadísticas de error:")
    print(f"Error medio: {np.mean(errors):.2f} UF")
    print(f"Error medio absoluto: {np.mean(np.abs(errors)):.2f} UF")
    print(f"Error porcentual medio: {np.mean(np.abs(error_percentage)):.2f}%")
    print(f"Desviación estándar del error: {np.std(errors):.2f} UF")

def guardar_modelo(best_model, scaler, label_encoders, available_features, optimized_analysis, kmeans_model=None):
    """Guardar el modelo, preprocesadores y métricas"""
    print("\n=== GUARDANDO MODELO Y MÉTRICAS ===")
    
    # Crear directorio encargos_data si no existe
    import os
    os.makedirs('encargos_data', exist_ok=True)
    
    # Guardar modelo y preprocesadores en el directorio encargos_data
    joblib.dump(best_model, 'encargos_data/modelo_valuacion_propiedades_v2.pkl')
    joblib.dump(scaler, 'encargos_data/scaler_valuacion_propiedades_v2.pkl')
    joblib.dump(label_encoders, 'encargos_data/label_encoders_valuacion_propiedades_v2.pkl')
    joblib.dump(available_features, 'encargos_data/features_valuacion_propiedades_v2.pkl')
    
    # Guardar modelo KMeans si está disponible
    if kmeans_model is not None:
        joblib.dump(kmeans_model, 'encargos_data/kmeans_zona_geografica_v2.pkl')
        print(" Modelo KMeans guardado en encargos_data/")
    
    # Guardar métricas en archivo JSON
    from datetime import datetime
    import json
    
    metricas = {
        'tipo_modelo': type(best_model).__name__,
        'r2_score': f"{optimized_analysis['test_r2']:.4f}",
        'rmse': f"{optimized_analysis['test_rmse']:.2f} UF",
        'mae': f"{optimized_analysis['test_mae']:.2f} UF",
        'overfitting_gap': f"{optimized_analysis['r2_gap']:.4f}",
        'data_leakage': 'Eliminado',
        'fecha_entrenamiento': datetime.now().isoformat(),
        'version_modelo': 'V2',
        'metricas_detalladas': {
            'train_r2': optimized_analysis['train_r2'],
            'test_r2': optimized_analysis['test_r2'],
            'train_rmse': optimized_analysis['train_rmse'],
            'test_rmse': optimized_analysis['test_rmse'],
            'train_mae': optimized_analysis['train_mae'],
            'test_mae': optimized_analysis['test_mae'],
            'r2_gap': optimized_analysis['r2_gap'],
            'rmse_gap': optimized_analysis['rmse_gap'],
            'mae_gap': optimized_analysis['mae_gap']
        }
    }
    
    with open('encargos_data/metricas_encargo.json', 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    
    print("Modelo, preprocesadores y métricas guardados exitosamente en encargos_data/!")

def generar_reporte_final(optimized_analysis, feature_importance, correlations):
    """Generar reporte final"""
    print("\n" + "="*60)
    print("REPORTE FINAL DEL MODELO V2")
    print("="*60)
    
    print(f"R² Score (Train): {optimized_analysis['train_r2']:.4f}")
    print(f"R² Score (Test): {optimized_analysis['test_r2']:.4f}")
    print(f"RMSE (Train): {optimized_analysis['train_rmse']:.2f} UF")
    print(f"RMSE (Test): {optimized_analysis['test_rmse']:.2f} UF")
    print(f"MAE (Train): {optimized_analysis['train_mae']:.2f} UF")
    print(f"MAE (Test): {optimized_analysis['test_mae']:.2f} UF")
    
    print(f"\nAnálisis de Overfitting:")
    print(f"R² gap: {optimized_analysis['r2_gap']:.4f}")
    print(f"RMSE gap: {optimized_analysis['rmse_gap']:.4f}")
    print(f"MAE gap: {optimized_analysis['mae_gap']:.4f}")
    
    if optimized_analysis['r2_gap'] > 0.1:
        print(" ADVERTENCIA: Posible overfitting detectado")
    elif optimized_analysis['r2_gap'] > 0.05:
        print(" ADVERTENCIA: Ligero overfitting detectado")
    else:
        print(" Modelo bien generalizado")
    
    if feature_importance is not None:
        print("\nTop 5 features más importantes:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    
    print("\nArchivos generados en encargos_data/:")
    print("- modelo_valuacion_propiedades_v2.pkl")
    print("- scaler_valuacion_propiedades_v2.pkl")
    print("- label_encoders_valuacion_propiedades_v2.pkl")
    print("- features_valuacion_propiedades_v2.pkl")
    print("- kmeans_zona_geografica_v2.pkl")
    print("- densidad_por_zona_v2.pkl")
    print("- metricas_encargo.json")

def main():
    """Función principal"""
    print("=== MODELO DE MACHINE LEARNING PARA VALUACIÓN DE PROPIEDADES V2 ===")
    
    # 1. Cargar datos
    df = cargar_datos()
    
    # 2. Eliminar outliers en años de construcción
    df_clean = eliminar_outliers_ano_construccion(df)
    
    # 3. EDA
    correlations = realizar_eda(df_clean)
    
    # 4. Feature Engineering (sin data leakage)
    df_fe, label_encoders = feature_engineering(df_clean)
    
    # 5. Preparar datos (sin data leakage)
    X_train, X_test, y_train, y_test, scaler, available_features = preparar_datos(df_fe)
    
    # 6. Análisis de data leakage
    data_leakage_correlations = analizar_data_leakage(df_fe, available_features)
    
    # 7. Entrenar modelos con análisis de overfitting
    modelos, df_overfitting = entrenar_modelos_con_analisis_overfitting(X_train, y_train, X_test, y_test)
    
    # 8. Optimizar mejor modelo sin overfitting
    best_model, optimized_analysis = optimizar_mejor_modelo_sin_overfitting(
        df_overfitting, modelos, X_train, y_train, X_test, y_test
    )
    
    # 9. Analizar importancia de features
    feature_importance = analizar_importancia_features(best_model, available_features)
    
    # 10. Analizar errores
    y_pred = best_model.predict(X_test)
    analizar_errores(y_test, y_pred)
    
    # 11. Guardar modelo
    guardar_modelo(best_model, scaler, label_encoders, available_features, optimized_analysis)
    
    # 12. Generar reporte final
    generar_reporte_final(optimized_analysis, feature_importance, data_leakage_correlations)

    # Guardar importancia de variables en JSON
    if feature_importance is not None:
        guardar_importancia_variables_json(feature_importance)

if __name__ == "__main__":
    main() 