import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os
import math

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Slider, CustomJS, ColorBar, FixedTicker, Button
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
        background_fill_color="white",
        x_range=(x_min, x_max),  # Fijar rango para el eje X
        y_range=(y_min, y_max)   # Fijar rango para el eje Y
    )

    # Usar `rect` con color basado en PROF_TON_y
    p.rect(
        x="XC", y="YC", 
        width=square_size, height=square_size,
        source=source_filtered, 
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
        ("X ref", "@X{0,0}"),
        ("Y ref", "@Y{0,0}"),
        ("Toneladas (bloque)", "@TONNES_x{0,0}"),
        ("Toneladas (cluster)", "@TONNES_y{0,0}"),
        ("Prof/Ton (bloque)", "@PROF_TON_x{0.00}"),
        ("Prof/Ton (cluster)", "@PROF_TON_y{0.00}"),
        ("Ley I2", "@LEY_I2{0,0.00}"),
        ("Ley NANO3", "@NANO3{0.00}"),
        ("Factor Caliche I2", "@FC_I2{0.00}")
    ])
    p.add_tools(hover)

    # Calcular 6 ticks equidistantes para la colorbar
    ticks = list(np.linspace(prof_min, prof_max, 6))

    # Crear la barra de color con orientación horizontal, tamaño reducido y ticks fijos
    color_bar = ColorBar(
        color_mapper=color_mapper['transform'],
        ticker=FixedTicker(ticks=ticks),
        label_standoff=12,
        orientation="horizontal",
        location=(0, 0),
        height=5,   # Altura (grosor) reducida
        width=round(width*0.8),  # Ajusta el ancho si es necesario
        title="Prof/Ton"
    )

    # Agrega la colorbar al gráfico, debajo de él
    p.add_layout(color_bar, 'below')

    # Redondear hacia el entero anterior más cercano para los valores mínimo y máximo
    min_value_rounded = math.floor(df_merge["PROF_TON_y"].min())
    max_value_rounded = math.floor(df_merge["PROF_TON_y"].max())
    # Crear el slider con los valores redondeados
    slider = Slider(
        start=min_value_rounded, 
        end=max_value_rounded, 
        value=min_value_rounded, 
        step=0.5, 
        align="center",
        title="Mínimo PROF_TON"
    )

    # Definir el callback para actualizar los datos y la visibilidad
    callback = CustomJS(args=dict(source=source, source_filtered=source_filtered, slider=slider), code="""
    var data = source.data;
    var fdata = {
        'XC': [],
        'YC': [],
        'CLUSTER_ID': [],
        'TONNES_x': [],
        'TONNES_y': [],
        'PROF_TON_x': [],
        'PROF_TON_y': [],
        'LEY_I2': [],
        'NANO3': [],
        'FC_I2': []
    };
    var min_value = slider.value;
    
    // Recorremos todos los datos y filtramos aquellos que cumplen la condición
    for (var i = 0; i < data['PROF_TON_y'].length; i++) {
        if (data['PROF_TON_y'][i] >= min_value) {
            fdata['XC'].push(data['XC'][i]);
            fdata['YC'].push(data['YC'][i]);
            fdata['CLUSTER_ID'].push(data['CLUSTER_ID'][i]);
            fdata['TONNES_x'].push(data['TONNES_x'][i]);
            fdata['TONNES_y'].push(data['TONNES_y'][i]);
            fdata['PROF_TON_x'].push(data['PROF_TON_x'][i]);
            fdata['PROF_TON_y'].push(data['PROF_TON_y'][i]);
            fdata['LEY_I2'].push(data['LEY_I2'][i]);
            fdata['NANO3'].push(data['NANO3'][i]);
            fdata['FC_I2'].push(data['FC_I2'][i]);
        }
    }
    
    // Se asigna el nuevo objeto de datos filtrados a la fuente
    source_filtered.data = fdata;
    """)

    # Vincular el slider con el callback
    slider.js_on_change("value", callback)

    # Botón para exportar a CSV
    export_button = Button(label="Exportar CSV", button_type="success")

    export_callback = CustomJS(args=dict(source=source_filtered), code="""
        // Función para convertir los datos de la fuente en formato CSV
        function table_to_csv(source) {
            const columns = Object.keys(source.data);
            const nrows = source.data[columns[0]].length;
            let lines = [];
            // Agrega el encabezado
            lines.push(columns.join(','));
            // Recorre cada fila
            for (let i = 0; i < nrows; i++) {
                let row = [];
                for (let j = 0; j < columns.length; j++) {
                    let column = columns[j];
                    row.push(source.data[column][i]);
                }
                lines.push(row.join(','));
            }
            return lines.join('\\n').concat('\\n');
        }
        
        // Nombre del archivo a exportar
        const filename = 'datos_filtrados.csv';
        let filetext = table_to_csv(source);
        
        // Crear un Blob con el contenido CSV y disparar la descarga
        let blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });
        if (navigator.msSaveBlob) { // Para IE 10+
            navigator.msSaveBlob(blob, filename);
        } else {
            let link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            link.target = "_blank";
            link.style.visibility = 'hidden';
            link.dispatchEvent(new MouseEvent('click'));
        }
    """)

    # Vincular el callback al botón
    export_button.js_on_click(export_callback)

    # Agregar el botón al layout junto con el gráfico y el slider
    layout = column(p, slider, export_button, align="center")

    # Mostrar el gráfico
    show(layout)

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
        background_fill_color="white",
        x_range=(x_min, x_max),  # Fijar rango para el eje X
        y_range=(y_min, y_max)   # Fijar rango para el eje Y
    )

    # Usar `rect` en lugar de `scatter` para mantener escala constante
    p.rect(
        x="XC", y="YC", 
        width=square_size, height=square_size,
        source=source_filtered, 
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

    # Redondear hacia el entero anterior más cercano para los valores mínimo y máximo
    min_value_rounded = math.floor(df_merge["PROF_TON_y"].min())
    max_value_rounded = math.floor(df_merge["PROF_TON_y"].max())
    # Crear el slider con los valores redondeados
    slider = Slider(
        start=min_value_rounded, 
        end=max_value_rounded, 
        value=min_value_rounded, 
        step=0.5, 
        title="Mínimo PROF_TON"
    )

    # Definir el callback para actualizar los datos y la visibilidad
    callback = CustomJS(args=dict(source=source, source_filtered=source_filtered, slider=slider), code="""
    var data = source.data;
    var fdata = {
        'XC': [],
        'YC': [],
        'CLUSTER_ID': [],
        'TONNES_x': [],
        'TONNES_y': [],
        'PROF_TON_x': [],
        'PROF_TON_y': [],
        'LEY_I2': [],
        'NANO3': [],
        'FC_I2': []
    };
    var min_value = slider.value;
    
    // Recorremos todos los datos y filtramos aquellos que cumplen la condición
    for (var i = 0; i < data['PROF_TON_y'].length; i++) {
        if (data['PROF_TON_y'][i] >= min_value) {
            fdata['XC'].push(data['XC'][i]);
            fdata['YC'].push(data['YC'][i]);
            fdata['CLUSTER_ID'].push(data['CLUSTER_ID'][i]);
            fdata['TONNES_x'].push(data['TONNES_x'][i]);
            fdata['TONNES_y'].push(data['TONNES_y'][i]);
            fdata['PROF_TON_x'].push(data['PROF_TON_x'][i]);
            fdata['PROF_TON_y'].push(data['PROF_TON_y'][i]);
            fdata['LEY_I2'].push(data['LEY_I2'][i]);
            fdata['NANO3'].push(data['NANO3'][i]);
            fdata['FC_I2'].push(data['FC_I2'][i]);
        }
    }
    
    // Se asigna el nuevo objeto de datos filtrados a la fuente
    source_filtered.data = fdata;
    """)

    # Vincular el slider con el callback
    slider.js_on_change("value", callback)

    # Crear layout con gráfico y slider
    layout = column(p, slider)

    # Mostrar el gráfico
    show(layout)