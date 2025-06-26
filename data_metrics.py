import pandas as pd
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()


def metricas_comuna():
    df=pd.read_parquet(os.environ['DATA_METRICS_FILE'])

    df_metrics=df.groupby('Comuna').agg(
        avg_price_uf=('precio', 'median'),
        superficie=('superficie_util', 'mean'),
        n_properties=('precio', 'count')).reset_index()

    return df_metrics