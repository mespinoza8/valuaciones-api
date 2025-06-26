import joblib
import json
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor


models = {
    'LightGBM': LGBMRegressor(random_state=42, n_jobs=-1),
    'CatBoost': CatBoostRegressor(verbose=0, random_state=42, thread_count=-1),
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1)
}


def preparar_datos_para_modelo(df: pd.DataFrame, target: str='precio'):
    X = df.drop(target, axis=1)
    y = df[target]
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer([('num', numeric_transformer, num_cols), ('cat', categorical_transformer, cat_cols)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, preprocessor


def evaluar_modelo_cv(X: pd.DataFrame, y: pd.Series, preprocessor, cv: int=5):

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        pipe = Pipeline([('preproc', preprocessor), ('model', model)])
        y_pred = cross_val_predict(pipe, X, y, cv=kf)
        mse = mean_squared_error(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        r2 = r2_score(y, y_pred)
        results[name] = {'rmse': np.sqrt(mse), 'r2': r2,'mape': mape}

    return results


def entrenar_y_guardar_modelo(df: pd.DataFrame,
                              model_path: str = None):

    # 1) Preparar datos, cross‐val, elegir mejor (igual que antes)…
    X_train, X_test, y_train, y_test, preproc = preparar_datos_para_modelo(df)
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])

    results = evaluar_modelo_cv(X_full, y_full, preproc)
    best = min(results, key=lambda m: results[m]['rmse'])

    # 2) Entrenar pipeline final
    final_pipe = Pipeline([
        ('preprocessor', preproc),
        ('model', models[best])
    ])
    final_pipe.fit(X_full, y_full)

    # 3) Determinar dónde guardar
    if model_path is None:
        # Directorio donde reside este script (model.py)
        project_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(project_dir, 'modelo_valoracion.pkl')

    # 4) Serializar
    joblib.dump(final_pipe, model_path)
    print(f"Modelo guardado en: {model_path}")

    # Exportar resultados (métricas) a JSON
    metrics_output = {
        "model_name": best,
        "metrics": results
    }
    # Guardar en el mismo directorio del modelo
    metrics_path = os.path.join(os.path.dirname(model_path), "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics_output, mf, indent=4, ensure_ascii=False)
    print(f"Métricas guardadas en: {metrics_path}")


    return results, best