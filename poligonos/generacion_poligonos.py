import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist

def eliminar_puntas(df, coordenadas_dict):
    coordenadas_filtradas = defaultdict(lambda: None)

    for (x, y), valor in coordenadas_dict.items():
        # Definir las condiciones correctamente
        cond1 = all([
            (x+1, y) in coordenadas_dict,
            (x+1, y+1) in coordenadas_dict,
            (x, y+1) in coordenadas_dict
        ])

        cond2 = all([
            (x+1, y) in coordenadas_dict,
            (x+1, y-1) in coordenadas_dict,
            (x, y-1) in coordenadas_dict
        ])

        cond3 = all([
            (x-1, y) in coordenadas_dict,
            (x-1, y+1) in coordenadas_dict,
            (x, y+1) in coordenadas_dict
        ])

        cond4 = all([
            (x-1, y) in coordenadas_dict,
            (x-1, y-1) in coordenadas_dict,
            (x, y-1) in coordenadas_dict
        ])

        # Si cumple alguna de las condiciones, mantener el punto
        if cond1 or cond2 or cond3 or cond4:
            coordenadas_filtradas[(x, y)] = valor
        else:
            df.loc[(df['X'] == x) & (df['Y'] == y), 'CLUSTER_ID'] = 1
    return coordenadas_filtradas

def expandir_clusters(df, inicial, coordenadas_dict, umbral_toneladas=100000):
    # Inicializar límites del cluster a partir del bloque inicial
    x0, y0 = inicial
    min_x = max_x = int(x0)
    min_y = max_y = int(y0)
    
    # Diccionario para guardar la información de los bloques del cluster
    coordenadas_cluster_dict = {}
    # Conjunto para guardar las coordenadas (el cluster)
    cluster_coords = set()
    
    # Agregar el bloque inicial
    cluster_coords.add(inicial)
    coordenadas_cluster_dict[inicial] = coordenadas_dict[inicial]
    suma_tonnes = coordenadas_dict[inicial]['TONNES']
    
    # Remover el bloque inicial del diccionario general para no volver a procesarlo
    del coordenadas_dict[inicial]
    
    # Bucle de expansión rectangular
    while suma_tonnes < umbral_toneladas:
        opciones = {}
        
        # --- Expansión hacia ARRIBA ---
        candidate1 = [(x, min_y - 1) for x in range(min_x, max_x + 1) if coordenadas_dict.get((x, min_y - 1)) is not None]
        candidate2 = [(x, min_y - 2) for x in range(min_x, max_x + 1) if coordenadas_dict.get((x, min_y - 2)) is not None]
        if candidate1:
            bloques_arriba = candidate1 + candidate2 if candidate2 else candidate1
            total_blocks = len(candidate1) + (len(candidate2) if candidate2 else 0)
            promedio_arriba = ((sum(coordenadas_dict[b]['PROF_TON'] for b in candidate1) +
                                (sum(coordenadas_dict[b]['PROF_TON'] for b in candidate2) if candidate2 else 0))
                               / total_blocks)
            opciones['arriba'] = (bloques_arriba, promedio_arriba, 2 if candidate2 else 1)
        
        # --- Expansión hacia ABAJO ---
        candidate1 = [(x, max_y + 1) for x in range(min_x, max_x + 1) if coordenadas_dict.get((x, max_y + 1)) is not None]
        candidate2 = [(x, max_y + 2) for x in range(min_x, max_x + 1) if coordenadas_dict.get((x, max_y + 2)) is not None]
        if candidate1:
            bloques_abajo = candidate1 + candidate2 if candidate2 else candidate1
            total_blocks = len(candidate1) + (len(candidate2) if candidate2 else 0)
            promedio_abajo = ((sum(coordenadas_dict[b]['PROF_TON'] for b in candidate1) +
                               (sum(coordenadas_dict[b]['PROF_TON'] for b in candidate2) if candidate2 else 0))
                              / total_blocks)
            opciones['abajo'] = (bloques_abajo, promedio_abajo, 2 if candidate2 else 1)
        
        # --- Expansión hacia IZQUIERDA ---
        candidate1 = [(min_x - 1, y) for y in range(min_y, max_y + 1) if coordenadas_dict.get((min_x - 1, y)) is not None]
        candidate2 = [(min_x - 2, y) for y in range(min_y, max_y + 1) if coordenadas_dict.get((min_x - 2, y)) is not None]
        if candidate1:
            bloques_izquierda = candidate1 + candidate2 if candidate2 else candidate1
            total_blocks = len(candidate1) + (len(candidate2) if candidate2 else 0)
            promedio_izquierda = ((sum(coordenadas_dict[b]['PROF_TON'] for b in candidate1) +
                                  (sum(coordenadas_dict[b]['PROF_TON'] for b in candidate2) if candidate2 else 0))
                                 / total_blocks)
            opciones['izquierda'] = (bloques_izquierda, promedio_izquierda, 2 if candidate2 else 1)
        
        # --- Expansión hacia DERECHA ---
        candidate1 = [(max_x + 1, y) for y in range(min_y, max_y + 1) if coordenadas_dict.get((max_x + 1, y)) is not None]
        candidate2 = [(max_x + 2, y) for y in range(min_y, max_y + 1) if coordenadas_dict.get((max_x + 2, y)) is not None]
        if candidate1:
            bloques_derecha = candidate1 + candidate2 if candidate2 else candidate1
            total_blocks = len(candidate1) + (len(candidate2) if candidate2 else 0)
            promedio_derecha = ((sum(coordenadas_dict[b]['PROF_TON'] for b in candidate1) +
                                 (sum(coordenadas_dict[b]['PROF_TON'] for b in candidate2) if candidate2 else 0))
                                / total_blocks)
            opciones['derecha'] = (bloques_derecha, promedio_derecha, 2 if candidate2 else 1)
        
        # Si no hay más bloques para expandir, salimos del bucle
        if not opciones:
            break
        
        # Seleccionar la dirección con mayor promedio de PROF_TON
        direccion_seleccionada = max(opciones, key=lambda d: opciones[d][1])
        bloques_a_agregar, _, thickness = opciones[direccion_seleccionada]
        
        # Agregar todos los bloques de la frontera seleccionada
        for bloque in bloques_a_agregar:
            suma_tonnes += coordenadas_dict[bloque]['TONNES']
            cluster_coords.add(bloque)
            coordenadas_cluster_dict[bloque] = coordenadas_dict[bloque]
            del coordenadas_dict[bloque]
            
        # Actualizar límites del cluster según la dirección seleccionada y el grosor agregado
        if direccion_seleccionada == 'arriba':
            min_y -= thickness
        elif direccion_seleccionada == 'abajo':
            max_y += thickness
        elif direccion_seleccionada == 'izquierda':
            min_x -= thickness
        elif direccion_seleccionada == 'derecha':
            max_x += thickness
    
    # Al finalizar la expansión, verificamos si la suma de TONNES excede 2 * umbral_toneladas.
    clusters_finales = []
    toneladas_finales = []
    if suma_tonnes > 2 * umbral_toneladas:
        clusters_divididos, toneladas_divididas = dividir_cluster(cluster_coords, 
                                                                  coordenadas_cluster_dict, 
                                                                  umbral_toneladas)
        clusters_finales.extend(clusters_divididos)
        toneladas_finales.extend(toneladas_divididas)
    else:
        clusters_finales.append(cluster_coords)
        toneladas_finales.append(suma_tonnes)
    
    return clusters_finales, toneladas_finales


def dividir_cluster(cluster, cluster_info, umbral_toneladas):
    """
    Recibe un cluster (conjunto de coordenadas) y su diccionario 'cluster_info' (con datos de cada bloque),
    y si el total de TONNES excede 2 * umbral_toneladas, lo divide en dos subclusters a lo largo del eje de simetría.
    
    Se selecciona el eje (vertical u horizontal) que favorezca una forma lo más cuadrada posible.
    La función se aplica de forma recursiva.
    
    Devuelve:
      - Una lista de clusters resultantes (cada uno es un conjunto de coordenadas).
      - Una lista con la suma de TONNES de cada cluster resultante.
    """
    # Calcular la suma total de TONNES del cluster actual
    suma = sum(cluster_info[b]['TONNES'] for b in cluster)
    if suma <= 2 * umbral_toneladas:
        return [cluster], [suma]
    
    # Determinar los límites del cluster
    xs = [b[0] for b in cluster]
    ys = [b[1] for b in cluster]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    # Dividir en dos subclusters según la dimensión dominante
    cluster1 = set()
    cluster2 = set()
    info1 = {}
    info2 = {}
    
    if width >= height:
        # División vertical (por eje X)
        x_split = (min_x + max_x) // 2
        for bloque in cluster:
            if bloque[0] <= x_split:
                cluster1.add(bloque)
                info1[bloque] = cluster_info[bloque]
            else:
                cluster2.add(bloque)
                info2[bloque] = cluster_info[bloque]
    else:
        # División horizontal (por eje Y)
        y_split = (min_y + max_y) // 2
        for bloque in cluster:
            if bloque[1] <= y_split:
                cluster1.add(bloque)
                info1[bloque] = cluster_info[bloque]
            else:
                cluster2.add(bloque)
                info2[bloque] = cluster_info[bloque]
    
    # Aplicar recursividad en cada subcluster (si es necesario)
    clusters_finales = []
    toneladas_finales = []
    for subcluster, subinfo in [(cluster1, info1), (cluster2, info2)]:
        sub_suma = sum(subinfo[b]['TONNES'] for b in subcluster)
        if sub_suma > 2 * umbral_toneladas:
            sub_clusters, sub_tonnes = dividir_cluster(subcluster, subinfo, umbral_toneladas)
            clusters_finales.extend(sub_clusters)
            toneladas_finales.extend(sub_tonnes)
        else:
            clusters_finales.append(subcluster)
            toneladas_finales.append(sub_suma)
    
    return clusters_finales, toneladas_finales

def encontrar_y_procesar_clusters(df, coordenadas_dict, umbral_toneladas=100000):
    """
    Función que encuentra y procesa todos los clusters:
      - Mientras haya bloques en 'coordenadas_dict', se toma uno (por ejemplo, el de mayor PROF_TON)
        y se expande (y, si es necesario, divide) formando uno o varios clusters.
      - Cada bloque asignado se retira de 'coordenadas_dict' para no procesarse nuevamente.
      - Finalmente, se actualiza el DataFrame (df) asignando un CLUSTER_ID único a cada cluster.
    
    Devuelve:
      - clusters_finales: Lista de clusters (cada uno es un conjunto de coordenadas).
      - toneladas_por_cluster: Lista con la suma de TONNES de cada cluster.
    """
    clusters_finales = []
    toneladas_por_cluster = []
    
    while coordenadas_dict:
        # Seleccionar el bloque inicial (por ejemplo, el de mayor PROF_TON)
        inicial = max(coordenadas_dict, key=lambda x: coordenadas_dict[x]['PROF_TON'])
        nuevos_clusters, nuevas_toneladas = expandir_clusters(df, inicial, coordenadas_dict, umbral_toneladas)
        clusters_finales.extend(nuevos_clusters)
        toneladas_por_cluster.extend(nuevas_toneladas)
    
    # Actualizar el DataFrame asignando CLUSTER_IDs
    # Se asume que la columna 'CLUSTER_ID' existe; de no ser así, se crea.
    cluster_id_actual = df['CLUSTER_ID'].max() + 1 if not df['CLUSTER_ID'].empty else 1
    for cluster in clusters_finales:
        for (x, y) in cluster:
            df.loc[(df['X'] == x) & (df['Y'] == y), 'CLUSTER_ID'] = cluster_id_actual
        cluster_id_actual += 1
    
    return df

def reasignar_puntos(df, umbral_toneladas=100000):
    """
    Reasigna puntos de clusters pequeños y del cluster 0 a clusters grandes.
    
    Para cada cluster pequeño se calcula, para cada cluster grande, el promedio de las distancias mínimas:
    es decir, para cada punto del cluster pequeño se toma la distancia al punto más cercano del cluster grande.
    Se selecciona como candidato a cluster receptor aquel cuya distancia promedio esté dentro de un factor 1.2 
    del mínimo obtenido, y en caso de haber varios, el que tenga menor TONNES acumulado.
    
    Finalmente, se reasignan todos los puntos del cluster pequeño al cluster grande elegido.
    """

    # Calcular el tonelaje total de cada cluster.
    tamanos_clusters = df.groupby('CLUSTER_ID')['TONNES'].sum().copy()

    # Definir clusters grandes y pequeños.
    clusters_grandes = tamanos_clusters[(tamanos_clusters >= umbral_toneladas) &
                                         (tamanos_clusters.index != 0) &
                                         (tamanos_clusters.index != 1)].index

    clusters_pequenos = tamanos_clusters[((tamanos_clusters < umbral_toneladas) | 
                                          (tamanos_clusters.index == 0)) &
                                          (tamanos_clusters.index != 1)].index

    # Filtrar los puntos de los clusters grandes.
    puntos_clusters_grandes = df[df['CLUSTER_ID'].isin(clusters_grandes)]
    if puntos_clusters_grandes.empty:
        return df

    # Precalcular las coordenadas de cada cluster grande.
    coord_grandes = {}
    for cluster in clusters_grandes:
        pts = df[df['CLUSTER_ID'] == cluster][['X', 'Y']].values.astype(np.float64)
        if pts.size > 0:
            coord_grandes[cluster] = pts

    # Iterar sobre cada cluster pequeño.
    for cluster_peq in clusters_pequenos:
        # Obtener los puntos del cluster pequeño actual.
        pts_peq = df[df['CLUSTER_ID'] == cluster_peq][['X', 'Y']].values.astype(np.float64)
        if pts_peq.size == 0:
            continue

        avg_distancias = {}
        # Para cada cluster grande, calcular el promedio de las distancias mínimas:
        for cluster_gr in clusters_grandes:
            pts_gr = coord_grandes.get(cluster_gr)
            if pts_gr is None or pts_gr.size == 0:
                continue
            # Calcular todas las distancias entre los puntos del pequeño y los puntos del grande.
            distancias = cdist(pts_peq, pts_gr)
            # Para cada punto del cluster pequeño se toma la distancia mínima (al punto más cercano del cluster grande).
            min_distancias = np.min(distancias, axis=1)
            # Promedio de las distancias mínimas para este par de clusters.
            avg = np.mean(min_distancias)
            avg_distancias[cluster_gr] = avg

        if not avg_distancias:
            continue

        # Encontrar el mínimo de las distancias promedio.
        min_avg = min(avg_distancias.values())

        # Seleccionar candidatos: aquellos clusters grandes cuya distancia promedio sea <= 1.2 * min_avg.
        candidatos = {k: v for k, v in avg_distancias.items() if v <= 1.2 * min_avg}
        if not candidatos:
            continue

        # De entre los candidatos, escoger el que tenga menor TONNES acumulado.
        candidato_final = min(candidatos.keys(), key=lambda x: tamanos_clusters[x])

        # Reasignar todos los puntos del cluster pequeño al cluster grande elegido.
        df.loc[df['CLUSTER_ID'] == cluster_peq, 'CLUSTER_ID'] = candidato_final

        # Actualizar el TONNES del cluster receptor (usando los datos actualizados en df).
        suma_toneladas = df[df['CLUSTER_ID'] == candidato_final]['TONNES'].sum()
        tamanos_clusters[candidato_final] = suma_toneladas

        # Actualizar las coordenadas del cluster receptor para futuros cálculos.
        coord_grandes[candidato_final] = df[df['CLUSTER_ID'] == candidato_final][['X', 'Y']].values.astype(np.float64)

    return df