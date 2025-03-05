from .utilidades_poligonos import preprocesar_datos, calcular_coordenadas_dict, calcular_metricas_por_cluster, actualizar_id_clusters
from .generacion_poligonos import eliminar_puntas, eliminar_bloques_indeseados, encontrar_y_procesar_clusters, reasignar_puntos
from .graficar_poligonos import graficar_iso_prof_ton_bloques_individuales, graficar_clusters, graficar_iso_prof_ton_clusters

import pandas as pd
import numpy as np
import os
from typing import List, Tuple


def generar_poligonos(
    ruta_csv: str, 
    ruta_outputs: str = "Resultados", 
    nombre_sector: str = "Mina", 
    limite_toneladas_formacion: int = 200_000, 
    limite_toneladas_disolucion: int = 100_000, 
    restricciones: List[str] = [], 
    ) -> None:
    
    tonelaje_string = f"{round(limite_toneladas_formacion / 1_000_000)}M" if limite_toneladas_formacion >= 1_000_000 else (f"{round(limite_toneladas_formacion / 1_000)}k" if limite_toneladas_formacion >= 1_000 else str(limite_toneladas_formacion))
    os.makedirs(ruta_outputs, exist_ok=True)

    df = pd.read_csv(ruta_csv)
    df = preprocesar_datos(df, restricciones)

    graficar_iso_prof_ton_bloques_individuales(df, ruta_outputs, nombre_sector)

    coordenadas_dict = calcular_coordenadas_dict(df)
    # FILTROS - Comentar en caso de querer desactivarlos!
    coordenadas_dict = eliminar_puntas(df, coordenadas_dict)
    coordenadas_dict = eliminar_bloques_indeseados(df, coordenadas_dict)

    df = encontrar_y_procesar_clusters(df, coordenadas_dict, limite_toneladas_formacion)
    df = reasignar_puntos(df, limite_toneladas_disolucion)
    df = actualizar_id_clusters(df)

    df_clusters = calcular_metricas_por_cluster(df)
    df = df.drop(columns=["I2 POND", "REC_I2 POND", "NANO3 POND", "REC_NANO3 POND", "PROF_TON_CLIPPED"])

    df.to_csv(os.path.join(ruta_outputs,f"{nombre_sector} CLUSTERS PUNTO A PUNTO ({tonelaje_string}).csv"), index=False)
    df_clusters.to_csv(os.path.join(ruta_outputs,f"{nombre_sector} INFO CLUSTERS ({tonelaje_string}).csv"), index=True)

    graficar_clusters(df, df_clusters, ruta_outputs, nombre_sector, tonelaje_string)
    graficar_iso_prof_ton_clusters(df, df_clusters, ruta_outputs, nombre_sector, tonelaje_string)