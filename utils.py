# utils.py
import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree


def convertir_precio(df: pd.DataFrame, valor_uf: float) -> pd.DataFrame:
    df = df.copy()
    mask_pesos = df['divisa'] == '$'
    df.loc[mask_pesos, 'precio'] = df.loc[mask_pesos, 'precio'] / valor_uf
    mask_usd = df['divisa'] == 'US$'
    if mask_usd.sum() > 0:
        tasa_usd_clp = 930
        df.loc[mask_usd, 'precio'] = (df.loc[mask_usd, 'precio'] * tasa_usd_clp) / valor_uf
    return df


def limpieza(valor):
    if pd.isna(valor) or valor == "":
        return np.nan
    if isinstance(valor, (int, float)):
        return valor
    match = re.search(r'(\d+(?:[\.,]\d+)?)', str(valor))
    if match:
        return float(match.group(1).replace(',', '.'))
    return np.nan


def recalcula_antiguedad(valor: float) -> float:
    return 2025 - valor if valor >= 1000 else valor


def rellenar_numerico_desde_desc(df: pd.DataFrame, col_desc: str, col_dest: str, pattern: str) -> pd.DataFrame:
    extraido = (df[col_desc].astype(str)
                   .str.extract(pattern, expand=False)
                   .pipe(pd.to_numeric, errors='coerce')
                   .round(0)
                   .astype('Int64'))
    orig = (df[col_dest]
            .replace('', pd.NA)
            .pipe(pd.to_numeric, errors='coerce')
            .round(0)
            .astype('Int64'))
    df[col_dest] = orig.fillna(extraido)
    return df


def rellenar_dormitorios(df: pd.DataFrame, desc: str='desc', dorm: str='dormitorios') -> pd.DataFrame:
    pattern = r'(?i)(\d+)\s*(?:dormitorios?|dorms?|habitaciones?)'
    return rellenar_numerico_desde_desc(df, desc, dorm, pattern)


def rellenar_estacionamientos(df: pd.DataFrame, desc: str='desc', est: str='estacionamientos') -> pd.DataFrame:
    pattern = r'(?i)(\d+)\s*(?:estacionamientos?|parking|estac\.?)'
    return rellenar_numerico_desde_desc(df, desc, est, pattern)


def preprocesar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    null_vals = ["", " ", "nan", "NaN", "null", "NULL", "na", "NA", "n/a", "N/A", "-", "none", "None"]
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace(null_vals, np.nan)
            df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x)
    return df


def geometry_points(df: pd.DataFrame, lon_col: str='longitud', lat_col: str='latitud') -> gpd.GeoDataFrame:
    geom = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    return gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326").reset_index(drop=True)


def calculate_nearest_distances(src: gpd.GeoDataFrame, tgt: gpd.GeoDataFrame) -> np.ndarray:
    src_coords = np.column_stack((src.geometry.x, src.geometry.y))
    tgt_coords = np.column_stack((tgt.geometry.x, tgt.geometry.y))
    tree = cKDTree(tgt_coords)
    dists, _ = tree.query(src_coords)
    R = 6371
    return dists * R * np.pi / 180


def calculate_nearest_distances_metro(src: gpd.GeoDataFrame, tgt: gpd.GeoDataFrame) -> np.ndarray:
    dists = []
    for p in src.geometry:
        dists.append(min(p.distance(line) for line in tgt.geometry))
    arr = np.array(dists)
    R = 6371
    return arr * R * np.pi / 180


def eliminar_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
    return df


def create_point_gdf(lat: float, lon: float, crs: str="EPSG:4326") -> gpd.GeoDataFrame:
    df = pd.DataFrame({'latitud':[lat], 'longitud':[lon]})
    geom = [Point(lon, lat)]
    return gpd.GeoDataFrame(df, geometry=geom, crs=crs)
