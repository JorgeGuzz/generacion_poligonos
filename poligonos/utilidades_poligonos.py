import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.interpolate import interp1d

# Ejemplo restricciones: ["ARQUEOLOGIA", "C_SAL", "EXPLOTADO", "INFRAESTRUCTURA", "LOMAS", "PILAS", "POLVORIN", "HALLAZGOS"]
# Ejemplo dimensiones : [25, 25, 0.5]
def preprocesar_datos(df, restricciones, dimensiones):
    #dim_x, dim_y, dim_z = dimensiones
    dim_x, dim_y = df['XINC'].iloc[0], df['YINC'].iloc[0]

    # Aplicamos las restricciones entregadas
    if "EIA" in df.columns:
        df = df[df["EIA"] == 1]
    
    for restriccion in restricciones:
        if restriccion in df.columns:
            df = df[df[restriccion] == 0]

    # Normalizamos en X e Y
    df["X"] = (df["XC"] - df["XC"].min()) / dim_x
    df["Y"] = (df["YC"] - df["YC"].min()) / dim_y

    # Agregamos columnas relevantes para el análisis posterior
    df["CLUSTER_ID"] = 0
    df['I2 POND'] = df['I2'] * df['TONNES']
    df['REC_I2 POND'] = df['REC_I2'] * df['TONNES']
    df['NANO3 POND'] = df['NANO3'] * df['TONNES']
    df['REC_NANO3 POND'] = df['REC_NANO3'] * df['TONNES']

    return df

# Genera un diccionario de posiciones para acelerar cálculos de adyacencia posteriores
def calcular_coordenadas_dict(df):
    coordenadas_dict = defaultdict(lambda: None)

    # Filtramos bloques con caracteristicas negativas
    df_filtrado = df[df['ESPESOR_CAL'] >= 2]
    df_filtrado = df[df['PROFIT'] >= 0]

    for _, row in df_filtrado.iterrows():
        x, y = row['X'], row['Y']
        # Crear un diccionario con el resto de las columnas
        valores = row.drop(['X', 'Y']).to_dict()
        # Asignar el diccionario a la coordenada (x, y)
        coordenadas_dict[(x, y)] = valores
    
    return coordenadas_dict

def calcular_metricas_por_cluster(df):
    df_clusters = df.groupby("CLUSTER_ID").agg(
        {
            "PROFIT": "sum",
            "TONNES": "sum",
            "I2 POND": "sum",
            "NANO3 POND": "sum",
            "REC_I2 POND": "sum",
            "REC_NANO3 POND": "sum",
            "ESPESOR_CAL": "sum",
            "CLUSTER_ID": "count",  # Contar bloques en cada cluster
            "XC": "mean",  # Calcular la media de las coordenadas X
            "YC": "mean"   # Calcular la media de las coordenadas Y
        }
    ).rename(columns={"CLUSTER_ID": "CANT_BLOQUES", "XC": "CENTROIDE_XC", "YC": "CENTROIDE_YC"})

    df_clusters['PROF_TON'] = df_clusters['PROFIT'] / df_clusters['TONNES']
    df_clusters['LEY_I2'] = df_clusters['I2 POND'] / df_clusters['TONNES']
    df_clusters['LEY_NANO3'] = df_clusters['NANO3 POND'] / df_clusters['TONNES']
    df_clusters['REC_I2'] = df_clusters['REC_I2 POND'] / df_clusters['TONNES']
    df_clusters['REC_NANO3'] = df_clusters['REC_NANO3 POND'] / df_clusters['TONNES']
    df_clusters['FC_I2'] = 1 / (df_clusters['LEY_I2'] * df_clusters['REC_I2']) * 1000

    return df_clusters

# Función para reasignar `CLUSTER_ID` para que sean continuos
def actualizar_id_clusters(df):
    CLUSTER_IDs = sorted(df['CLUSTER_ID'].unique())
    nuevo_id_map = {old_id: new_id for new_id, old_id in enumerate(CLUSTER_IDs) if old_id != 0}
    
    df['CLUSTER_ID'] = df['CLUSTER_ID'].map(lambda x: nuevo_id_map.get(x, 0))
    
    return df