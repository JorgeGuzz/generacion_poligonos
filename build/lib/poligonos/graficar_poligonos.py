import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Slider, CustomJS
from bokeh.layouts import column
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Category20

def graficar_iso_prof_ton_bloques_individuales(df, ruta_outputs, nombre_sector):
    x_min, x_max = df["XC"].min(), df["XC"].max()
    y_min, y_max = df["YC"].min(), df["YC"].max()
    aspect_ratio = (y_max - y_min) / (x_max - x_min)
    plot_width = 40
    plot_height = int(plot_width * aspect_ratio)

    # Calcular percentiles para manejar outliers
    prof_p1 = np.percentile(df["PROF_TON"], 1)   # Percentil 1
    prof_p99 = np.percentile(df["PROF_TON"], 99)  # Percentil 99

    # Clipping de los datos de PROF_TON dentro del rango de percentiles
    df["PROF_TON_CLIPPED"] = np.clip(df["PROF_TON"], prof_p1, prof_p99)

    plt.figure(figsize=(plot_width, plot_height))
    scatter_plot = sns.scatterplot(
        data=df,
        x="XC",
        y="YC",
        s=40,
        linestyle="-",
        linewidth=1,
        edgecolor="black",
        marker="s",
        hue="PROF_TON_CLIPPED",  # Usamos CLUSTER_ID para determinar los colores
        palette="gist_rainbow",    # Paleta categórica
        alpha=1,         # Transparencia
        legend=None      # Mostrar la leyenda de clusters
    )

    # Ajustar los límites de los ejes según los datos
    plt.xlim(df["XC"].min() - 25, df["XC"].max() + 25)
    plt.ylim(df["YC"].min() - 25, df["YC"].max() + 25)

    # Definir los intervalos de la grilla (ajusta el paso según necesites)
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    x_ticks = np.arange(x_min, x_max, step=125)  # Ajusta el step para cambiar el tamaño de los cuadrados
    y_ticks = np.arange(y_min, y_max, step=125)  

    plt.xticks(x_ticks, fontsize=6)
    plt.yticks(y_ticks, fontsize=6)

    # Configurar la grilla con más visibilidad
    plt.grid(
        True, 
        linestyle="--",  # Línea punteada
        linewidth=1,   # Grosor más grande
        color="gray",    # Color más visible
        alpha=0.8        # Opacidad más alta
    )

    # Configuración del colorbar con tamaño ajustado
    norm = plt.Normalize(prof_p1, prof_p99)
    sm = plt.cm.ScalarMappable(cmap="gist_rainbow", norm=norm)
    sm.set_array([])
    # Ajustar colorbar: más delgado y con etiquetas más grandes
    cbar = plt.colorbar(sm, ax=scatter_plot, pad=0.02, aspect=30, shrink=0.6)
    cbar.set_label("Profit / Tonelada", fontsize=50)
    cbar.ax.tick_params(labelsize=20)  # Aumentar tamaño de etiquetas

    # Configuraciones adicionales del gráfico
    plt.title(f"Distribución de Profit/Tonelada ({nombre_sector})", fontsize=50)
    plt.xlabel("XC", fontsize=20)
    plt.ylabel("YC", fontsize=20)

    # Mostrar el gráfico
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_outputs, f'Profit por tonelada bloque a bloque {nombre_sector}.png'), dpi=300, bbox_inches="tight")  # Guardar con alta resolución y sin bordes adicionales

def graficar_iso_prof_ton_clusters(df, df_clusters, ruta_outputs, nombre_sector, tonelaje_string):
    # Unir df con df_clusters para agregar las características de cada cluster
    df_merge = df.merge(df_clusters, on="CLUSTER_ID", how="left")

    # Convertir CLUSTER_ID a string para asignación de colores
    df_merge["CLUSTER_ID"] = df_merge["CLUSTER_ID"].astype(str)

    # Crear el ColumnDataSource con todas las columnas necesarias
    source = ColumnDataSource(df_merge)
    source_filtered = ColumnDataSource(df_merge)  # Fuente filtrada que se actualizará dinámicamente

    square_size = 23
    # Calcular límites y tamaño dinámico
    x_min, x_max = df_merge["XC"].min(), df_merge["XC"].max()
    y_min, y_max = df_merge["YC"].min(), df_merge["YC"].max()
    aspect_ratio = (y_max - y_min) / (x_max - x_min)
    width = 800
    height = int(width * aspect_ratio)
    height = max(800, min(height, 2000))

    # Definir el archivo HTML de salida
    output_path = os.path.join(ruta_outputs, f"{nombre_sector} - Clusters Interactivos ({tonelaje_string}) (Heatmap).html")
    output_file(output_path)
    # Obtener valores mínimo y máximo de PROF_TON_y para normalizar la escala de colores
    prof_min = np.percentile(df_merge["PROF_TON_y"], 1)   # Percentil 1
    prof_max = np.percentile(df_merge["PROF_TON_y"], 99)  # Percentil 99

    # Definir el número de colores en la paleta
    num_colors = 256  # Puedes reducirlo si lo deseas

    # Obtener la paleta gist_rainbow desde matplotlib y convertirla a hexadecimal
    gist_rainbow_palette = [
        mcolors.to_hex(cm.gist_rainbow(i / (num_colors - 1))) for i in range(num_colors)
    ]

    # Crear la asignación de colores con gist_rainbow
    color_mapper = linear_cmap(
        field_name="PROF_TON_y", 
        palette=gist_rainbow_palette, 
        low=prof_min, 
        high=prof_max
    )

    # Crear la figura de Bokeh
    p = figure(
        title=f"Distribución de Clusters {nombre_sector} ({tonelaje_string}) (Heatmap)", 
        width=width, 
        height=height,
        tools="pan, wheel_zoom, reset, poly_select",
        background_fill_color="white"
    )

    # Usar `rect` con color basado en PROF_TON_y
    p.rect(
        x="XC", y="YC", 
        width=square_size, height=square_size,
        source=source, 
        fill_alpha=0.9, 
        color=color_mapper,  # Se aplica el mapa de colores basado en PROF_TON_y
        line_color="black", 
        line_width=0.7
    )

    # Agregar tooltips con la información por cluster
    hover = HoverTool(tooltips=[
        ("Cluster", "@CLUSTER_ID"),
        ("XC", "@XC{0,0}"),
        ("YC", "@YC{0,0}"),
        ("Toneladas (bloque)", "@TONNES_x{0,0}"),
        ("Toneladas (cluster)", "@TONNES_y{0,0}"),
        ("Prof/Ton (bloque)", "@PROF_TON_x{0.00}"),
        ("Prof/Ton (cluster)", "@PROF_TON_y{0.00}"),
        ("Ley I2", "@LEY_I2{0,0.00}"),
        ("Ley NANO3", "@NANO3{0.00}"),
        ("Factor Caliche I2", "@FC_I2{0.00}")
    ])
    p.add_tools(hover)

    # Crear un slider para PROF_TON_y
    slider = Slider(start=df_merge["PROF_TON_y"].min(), 
                    end=df_merge["PROF_TON_y"].max(), 
                    value=df_merge["PROF_TON_y"].min(), 
                    step=0.01, 
                    title="Mínimo PROF_TON_y")

    # Definir el callback para actualizar los datos
    callback = CustomJS(args=dict(source=source, source_filtered=source_filtered, slider=slider), code="""
        var data = source.data;
        var data_filtered = source_filtered.data;
        var min_value = slider.value;

        data_filtered['XC'] = [];
        data_filtered['YC'] = [];
        data_filtered['CLUSTER_ID'] = [];
        data_filtered['TONNES_x'] = [];
        data_filtered['TONNES_y'] = [];
        data_filtered['PROF_TON_x'] = [];
        data_filtered['PROF_TON_y'] = [];
        data_filtered['LEY_I2'] = [];
        data_filtered['NANO3'] = [];
        data_filtered['FC_I2'] = [];

        for (var i = 0; i < data['PROF_TON_y'].length; i++) {
            if (data['PROF_TON_y'][i] >= min_value) {
                data_filtered['XC'].push(data['XC'][i]);
                data_filtered['YC'].push(data['YC'][i]);
                data_filtered['CLUSTER_ID'].push(data['CLUSTER_ID'][i]);
                data_filtered['TONNES_x'].push(data['TONNES_x'][i]);
                data_filtered['TONNES_y'].push(data['TONNES_y'][i]);
                data_filtered['PROF_TON_x'].push(data['PROF_TON_x'][i]);
                data_filtered['PROF_TON_y'].push(data['PROF_TON_y'][i]);
                data_filtered['LEY_I2'].push(data['LEY_I2'][i]);
                data_filtered['NANO3'].push(data['NANO3'][i]);
                data_filtered['FC_I2'].push(data['FC_I2'][i]);
            }
        }
        source_filtered.change.emit();
    """)

    # Vincular el slider con el callback
    slider.js_on_change("value", callback)

    # Crear layout con gráfico y slider
    layout = column(p, slider)

    # Mostrar el gráfico
    show(p)

def graficar_clusters(df, df_clusters, ruta_outputs, nombre_sector, tonelaje_string):
    # Definir el archivo HTML de salida
    output_path = os.path.join(ruta_outputs, f"{nombre_sector} - Clusters Interactivos ({tonelaje_string}).html")
    output_file(output_path)

    # Unir df con df_clusters para agregar las características de cada cluster
    df_merge = df.merge(df_clusters, on="CLUSTER_ID", how="left")

    # Convertir CLUSTER_ID a string para asignación de colores
    df_merge["CLUSTER_ID"] = df_merge["CLUSTER_ID"].astype(str)

    # Obtener lista única de clusters y paleta de colores
    unique_clusters = df_merge["CLUSTER_ID"].unique().tolist()
    num_clusters = len(unique_clusters)
    palette = Category20[20] if num_clusters <= 20 else Category20[20] * (num_clusters // 20 + 1)

    # Crear el ColumnDataSource con todas las columnas necesarias
    source = ColumnDataSource(df_merge)
    source_filtered = ColumnDataSource(df_merge)  # Fuente filtrada que se actualizará dinámicamente

    square_size = 23
    # Calcular límites y tamaño dinámico
    x_min, x_max = df_merge["XC"].min(), df_merge["XC"].max()
    y_min, y_max = df_merge["YC"].min(), df_merge["YC"].max()
    aspect_ratio = (y_max - y_min) / (x_max - x_min)
    width = 800
    height = int(width * aspect_ratio)
    height = max(800, min(height, 2000))

    # Crear la figura de Bokeh
    p = figure(
        title=f"Distribución de Clusters {nombre_sector} ({tonelaje_string})", 
        width=width, 
        height=height,
        tools="pan, wheel_zoom, reset, save",
        background_fill_color="white"
    )

    # Usar `rect` en lugar de `scatter` para mantener escala constante
    p.rect(
        x="XC", y="YC", 
        width=square_size, height=square_size,
        source=source, 
        fill_alpha=0.9, 
        color=factor_cmap("CLUSTER_ID", palette=palette, factors=unique_clusters),
        line_color="black", 
        line_width=0.7
    )

    # Agregar tooltips con información detallada del cluster
    hover = HoverTool(tooltips=[
        ("Cluster", "@CLUSTER_ID"),
        ("XC", "@XC{0,0}"),
        ("YC", "@YC{0,0}"),
        ("Toneladas (bloque)", "@TONNES_x{0,0}"),
        ("Toneladas (cluster)", "@TONNES_y{0,0}"),
        ("Prof/Ton (bloque)", "@PROF_TON_x{0.00}"),
        ("Prof/Ton (cluster)", "@PROF_TON_y{0.00}"),
        ("Ley I2", "@LEY_I2{0,0.00}"),
        ("Ley NANO3", "@NANO3{0.00}"),
        ("Factor Caliche I2", "@FC_I2{0.00}")
    ])
    p.add_tools(hover)

    # Crear un slider para PROF_TON_y
    slider = Slider(start=df_merge["PROF_TON_y"].min(), 
                    end=df_merge["PROF_TON_y"].max(), 
                    value=df_merge["PROF_TON_y"].min(), 
                    step=0.01, 
                    title="Mínimo PROF_TON_y")

    # Definir el callback para actualizar los datos
    callback = CustomJS(args=dict(source=source, source_filtered=source_filtered, slider=slider), code="""
        var data = source.data;
        var data_filtered = source_filtered.data;
        var min_value = slider.value;

        data_filtered['XC'] = [];
        data_filtered['YC'] = [];
        data_filtered['CLUSTER_ID'] = [];
        data_filtered['TONNES_x'] = [];
        data_filtered['TONNES_y'] = [];
        data_filtered['PROF_TON_x'] = [];
        data_filtered['PROF_TON_y'] = [];
        data_filtered['LEY_I2'] = [];
        data_filtered['NANO3'] = [];
        data_filtered['FC_I2'] = [];

        for (var i = 0; i < data['PROF_TON_y'].length; i++) {
            if (data['PROF_TON_y'][i] >= min_value) {
                data_filtered['XC'].push(data['XC'][i]);
                data_filtered['YC'].push(data['YC'][i]);
                data_filtered['CLUSTER_ID'].push(data['CLUSTER_ID'][i]);
                data_filtered['TONNES_x'].push(data['TONNES_x'][i]);
                data_filtered['TONNES_y'].push(data['TONNES_y'][i]);
                data_filtered['PROF_TON_x'].push(data['PROF_TON_x'][i]);
                data_filtered['PROF_TON_y'].push(data['PROF_TON_y'][i]);
                data_filtered['LEY_I2'].push(data['LEY_I2'][i]);
                data_filtered['NANO3'].push(data['NANO3'][i]);
                data_filtered['FC_I2'].push(data['FC_I2'][i]);
            }
        }
        source_filtered.change.emit();
    """)

    # Vincular el slider con el callback
    slider.js_on_change("value", callback)

    # Crear layout con gráfico y slider
    layout = column(p, slider)

    # Mostrar el gráfico
    show(p)