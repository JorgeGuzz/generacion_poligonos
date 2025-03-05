import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os
import math

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Slider, CustomJS, ColorBar, FixedTicker, Button, MultiSelect
from bokeh.layouts import column, row
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Category20

def graficar_iso_prof_ton_bloques_individuales(df, ruta_outputs, nombre_sector):

    # Cálculo de límites y razón de aspecto según tus datos
    x_min, x_max = df["XC"].min(), df["XC"].max()
    y_min, y_max = df["YC"].min(), df["YC"].max()
    aspect_ratio = (y_max - y_min) / (x_max - x_min)
    dim_x, dim_y = df['XINC'].iloc[0], df['YINC'].iloc[0]
    
    plot_height = 40
    plot_width = int(plot_height / aspect_ratio)

    # Recorte de outliers para la variable PROF_TON
    prof_p1 = np.percentile(df["PROF_TON"], 1)
    prof_p99 = np.percentile(df["PROF_TON"], 99)
    df["PROF_TON_CLIPPED"] = np.clip(df["PROF_TON"], prof_p1, prof_p99)

    # Crear la figura con las dimensiones originales calculadas
    fig = plt.figure(figsize=(plot_width, plot_height))
    ax = plt.gca()

    # Definir límites de los ejes con un margen extra
    margin = dim_x
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    # Forzar el renderizado para que se actualicen las transformaciones
    fig.canvas.draw()

    # Definir el step de la grilla (en unidades de datos)
    cell_size = dim_x

    # Calcular el tamaño en puntos de una celda en la dirección X
    p0_x = ax.transData.transform((0, 0))
    p1_x = ax.transData.transform((cell_size, 0))
    cell_width_points = np.abs(p1_x[0] - p0_x[0])

    # Calcular el tamaño en puntos de una celda en la dirección Y
    p0_y = ax.transData.transform((0, 0))
    p1_y = ax.transData.transform((0, cell_size))
    cell_height_points = np.abs(p1_y[1] - p0_y[1])

    # Usar el mínimo para asegurar que el cuadrado quepa en la grilla
    cell_size_points = min(cell_width_points, cell_height_points)
    # Por ejemplo, usar el 80% del tamaño de la celda para el cuadrado
    marker_side = 0.65 * cell_size_points
    # s se define como área en puntos²
    marker_size = marker_side ** 2

    # Graficar con scatterplot de Seaborn usando el tamaño calculado
    scatter_plot = sns.scatterplot(
        data=df,
        x="XC",
        y="YC",
        ax=ax,
        s=marker_size,
        marker="s",
        edgecolor="black",
        linewidth=marker_size*0.02 / (dim_x/25),
        hue="PROF_TON_CLIPPED",
        palette="gist_rainbow",
        alpha=1,
        legend=None
    )

    # Configurar los ticks y la grilla
    grid_step = cell_size * 25
    x_ticks = np.arange(x_min - margin, x_max + margin + grid_step, step=grid_step)
    y_ticks = np.arange(y_min - margin, y_max + margin + grid_step, step=grid_step)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(labelsize=6)
    ax.grid(True, linestyle="--", linewidth=1, color="gray", alpha=0.8)

    # Configurar el colorbar
    norm = plt.Normalize(prof_p1, prof_p99)
    sm = plt.cm.ScalarMappable(cmap="gist_rainbow", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30, shrink=0.6)
    cbar.set_label("Profit / Tonelada", fontsize=50)
    cbar.ax.tick_params(labelsize=20)

    # Título y etiquetas
    ax.set_title(f"Distribución de Profit/Tonelada ({nombre_sector})", fontsize=50)
    ax.set_xlabel("XC", fontsize=20)
    ax.set_ylabel("YC", fontsize=20)

    plt.tight_layout()
    plt.savefig(os.path.join(ruta_outputs, f'Profit por tonelada bloque a bloque {nombre_sector}.png'), dpi=300, bbox_inches="tight")

def graficar_iso_prof_ton_clusters(df, df_clusters, ruta_outputs, nombre_sector, tonelaje_string):
    # Unir df con df_clusters para agregar las características de cada cluster
    df_merge = df.merge(df_clusters, on="CLUSTER_ID", how="left")

    # Convertir CLUSTER_ID a string para asignación de colores
    df_merge["CLUSTER_ID"] = df_merge["CLUSTER_ID"].astype(str)

    # Crear el ColumnDataSource con todas las columnas necesarias
    source = ColumnDataSource(df_merge)
    source_filtered = ColumnDataSource(df_merge)  # Fuente filtrada que se actualizará dinámicamente

    # Fuente de datos estática con el DataFrame original "df" (con las columnas originales)
    source_static = ColumnDataSource(df)

    dim_x, dim_y = df['XINC'].iloc[0], df['YINC'].iloc[0]
    square_size = dim_x - (2 * dim_x / 25)
    # Calcular límites y tamaño dinámico
    x_min, x_max = df_merge["XC"].min(), df_merge["XC"].max()
    y_min, y_max = df_merge["YC"].min(), df_merge["YC"].max()
    aspect_ratio = (y_max - y_min) / (x_max - x_min)
    height = 500
    # height = max(800, min(height, 2000))
    width = int(height / aspect_ratio)

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
        ("Espesor Caliche (bloque)", "@ESPESOR_CAL_x{0.00}"),
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

    # Crear el widget MultiSelect con los clusters disponibles
    unique_clusters = df_merge["CLUSTER_ID"].unique().tolist()
    unique_clusters_sorted = sorted(unique_clusters, key=lambda x: int(x))
    multi_select = MultiSelect(
        title="Selecciona Clusters:",
        value=unique_clusters_sorted,   # Por defecto, se seleccionan todos
        options=unique_clusters_sorted,
        size=25
    )

    callback_unificado = CustomJS(
        args=dict(source=source, source_filtered=source_filtered, slider=slider, multi_select=multi_select),
        code="""
        var data = source.data;
        // Se define el objeto fdata con todas las claves que se esperan en source_filtered.
        var fdata = { 
            'XC': [], 
            'YC': [], 
            'X': [], 
            'Y': [], 
            'CLUSTER_ID': [],
            'TONNES_x': [], 
            'TONNES_y': [], 
            'PROF_TON_x': [], 
            'PROF_TON_y': [],
            'ESPESOR_CAL_x': [], 
            'LEY_I2': [], 
            'NANO3': [], 
            'FC_I2': []
        };
        
        var slider_val = slider.value;
        var selected_clusters = multi_select.value;
        
        // Recorrer todas las filas y aplicar ambos filtros:
        //   - El cluster debe estar entre los seleccionados.
        //   - PROF_TON_y debe ser mayor o igual al valor del slider.
        for (var i = 0; i < data['CLUSTER_ID'].length; i++) {
            if (selected_clusters.includes(data['CLUSTER_ID'][i]) && data['PROF_TON_y'][i] >= slider_val) {
                fdata['XC'].push(data['XC'][i]);
                fdata['YC'].push(data['YC'][i]);
                if(data.hasOwnProperty('X')) { fdata['X'].push(data['X'][i]); }
                if(data.hasOwnProperty('Y')) { fdata['Y'].push(data['Y'][i]); }
                fdata['CLUSTER_ID'].push(data['CLUSTER_ID'][i]);
                fdata['TONNES_x'].push(data['TONNES_x'][i]);
                fdata['TONNES_y'].push(data['TONNES_y'][i]);
                fdata['PROF_TON_x'].push(data['PROF_TON_x'][i]);
                fdata['PROF_TON_y'].push(data['PROF_TON_y'][i]);
                fdata['ESPESOR_CAL_x'].push(data['ESPESOR_CAL_x'][i]);
                fdata['LEY_I2'].push(data['LEY_I2'][i]);
                fdata['NANO3'].push(data['NANO3'][i]);
                fdata['FC_I2'].push(data['FC_I2'][i]);
            }
        }
        
        source_filtered.data = fdata;
        source_filtered.change.emit();
        """
    )

    slider.js_on_change("value", callback_unificado)
    multi_select.js_on_change("value", callback_unificado)

    # Botón para exportar CSV
    export_button = Button(label="Exportar CSV", button_type="success")

    # Callback JavaScript que filtra "source_static" usando los CLUSTER_ID presentes en "source_filtered"
    export_callback = CustomJS(
        args=dict(source_static=source_static, source_filtered=source_filtered),
        code="""
            // Función que convierte un objeto de datos en CSV
            function table_to_csv(data_dict) {
                const columns = Object.keys(data_dict);
                const nrows = data_dict[columns[0]].length;
                let lines = [];
                // Agregar cabecera
                lines.push(columns.join(','));
                // Agregar cada fila
                for (let i = 0; i < nrows; i++){
                    let row = [];
                    for (let j = 0; j < columns.length; j++){
                        row.push(data_dict[columns[j]][i]);
                    }
                    lines.push(row.join(','));
                }
                return lines.join('\\n').concat('\\n');
            }
            
            // Obtener los CLUSTER_ID de los datos filtrados (source_filtered)
            const filtered_data = source_filtered.data;
            const filtered_clusters = new Set(filtered_data['CLUSTER_ID']);
            
            // Obtener los datos originales (source_static) con las columnas de df
            const orig = source_static.data;
            let export_data = {};
            // Inicializar export_data con las mismas columnas que orig
            for (const key in orig) {
                export_data[key] = [];
            }
            
            const n = orig['CLUSTER_ID'].length;
            // Filtrar: incluir solo filas cuyo CLUSTER_ID (convertido a string) esté en el conjunto filtrado
            for (let i = 0; i < n; i++){
                if (filtered_clusters.has(String(orig['CLUSTER_ID'][i]))){
                    for (const key in orig){
                        export_data[key].push(orig[key][i]);
                    }
                }
            }
            
            // Convertir export_data a CSV
            const csv = table_to_csv(export_data);
            const filename = 'filtrado.csv';
            
            // Crear Blob y disparar la descarga del CSV
            let blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            if (navigator.msSaveBlob){
                navigator.msSaveBlob(blob, filename);
            } else {
                let link = document.createElement("a");
                link.href = URL.createObjectURL(blob);
                link.download = filename;
                link.target = "_blank";
                link.style.visibility = 'hidden';
                link.dispatchEvent(new MouseEvent('click'));
            }
        """
    )

    export_button.js_on_click(export_callback)

    # Crear layout
    layout = row(column(p, slider, align="center"), column(multi_select, export_button), align="center")

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

    # Fuente de datos estática con el DataFrame original "df" (con las columnas originales)
    source_static = ColumnDataSource(df)

    dim_x, dim_y = df['XINC'].iloc[0], df['YINC'].iloc[0]
    square_size = dim_x - (2 * dim_x / 25)
    # Calcular límites y tamaño dinámico
    x_min, x_max = df_merge["XC"].min(), df_merge["XC"].max()
    y_min, y_max = df_merge["YC"].min(), df_merge["YC"].max()
    aspect_ratio = (y_max - y_min) / (x_max - x_min)
    height = 500
    # height = max(800, min(height, 2000))
    width = int(height / aspect_ratio)

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
        ("X ref", "@X{0,0}"),
        ("Y ref", "@Y{0,0}"),
        ("Toneladas (bloque)", "@TONNES_x{0,0}"),
        ("Toneladas (cluster)", "@TONNES_y{0,0}"),
        ("Prof/Ton (bloque)", "@PROF_TON_x{0.00}"),
        ("Prof/Ton (cluster)", "@PROF_TON_y{0.00}"),
        ("Espesor Caliche (bloque)", "@ESPESOR_CAL_x{0.00}"),
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

    # Crear el widget MultiSelect con los clusters disponibles
    unique_clusters_sorted = sorted(unique_clusters, key=lambda x: int(x))
    multi_select = MultiSelect(
        title="Selecciona Clusters:",
        value=unique_clusters_sorted,   # Por defecto, se seleccionan todos
        options=unique_clusters_sorted,
        size=25
    )

    callback_unificado = CustomJS(
        args=dict(source=source, source_filtered=source_filtered, slider=slider, multi_select=multi_select),
        code="""
        var data = source.data;
        // Se define el objeto fdata con todas las claves que se esperan en source_filtered.
        var fdata = { 
            'XC': [], 
            'YC': [], 
            'X': [], 
            'Y': [], 
            'CLUSTER_ID': [],
            'TONNES_x': [], 
            'TONNES_y': [], 
            'PROF_TON_x': [], 
            'PROF_TON_y': [],
            'ESPESOR_CAL_x': [], 
            'LEY_I2': [], 
            'NANO3': [], 
            'FC_I2': []
        };
        
        var slider_val = slider.value;
        var selected_clusters = multi_select.value;
        
        // Recorrer todas las filas y aplicar ambos filtros:
        //   - El cluster debe estar entre los seleccionados.
        //   - PROF_TON_y debe ser mayor o igual al valor del slider.
        for (var i = 0; i < data['CLUSTER_ID'].length; i++) {
            if (selected_clusters.includes(data['CLUSTER_ID'][i]) && data['PROF_TON_y'][i] >= slider_val) {
                fdata['XC'].push(data['XC'][i]);
                fdata['YC'].push(data['YC'][i]);
                if(data.hasOwnProperty('X')) { fdata['X'].push(data['X'][i]); }
                if(data.hasOwnProperty('Y')) { fdata['Y'].push(data['Y'][i]); }
                fdata['CLUSTER_ID'].push(data['CLUSTER_ID'][i]);
                fdata['TONNES_x'].push(data['TONNES_x'][i]);
                fdata['TONNES_y'].push(data['TONNES_y'][i]);
                fdata['PROF_TON_x'].push(data['PROF_TON_x'][i]);
                fdata['PROF_TON_y'].push(data['PROF_TON_y'][i]);
                fdata['ESPESOR_CAL_x'].push(data['ESPESOR_CAL_x'][i]);
                fdata['LEY_I2'].push(data['LEY_I2'][i]);
                fdata['NANO3'].push(data['NANO3'][i]);
                fdata['FC_I2'].push(data['FC_I2'][i]);
            }
        }
        
        source_filtered.data = fdata;
        source_filtered.change.emit();
        """
    )

    slider.js_on_change("value", callback_unificado)
    multi_select.js_on_change("value", callback_unificado)

    # Botón para exportar CSV
    export_button = Button(label="Exportar CSV", button_type="success")

    # Callback JavaScript que filtra "source_static" usando los CLUSTER_ID presentes en "source_filtered"
    export_callback = CustomJS(
        args=dict(source_static=source_static, source_filtered=source_filtered),
        code="""
            // Función que convierte un objeto de datos en CSV
            function table_to_csv(data_dict) {
                const columns = Object.keys(data_dict);
                const nrows = data_dict[columns[0]].length;
                let lines = [];
                // Agregar cabecera
                lines.push(columns.join(','));
                // Agregar cada fila
                for (let i = 0; i < nrows; i++){
                    let row = [];
                    for (let j = 0; j < columns.length; j++){
                        row.push(data_dict[columns[j]][i]);
                    }
                    lines.push(row.join(','));
                }
                return lines.join('\\n').concat('\\n');
            }
            
            // Obtener los CLUSTER_ID de los datos filtrados (source_filtered)
            const filtered_data = source_filtered.data;
            const filtered_clusters = new Set(filtered_data['CLUSTER_ID']);
            
            // Obtener los datos originales (source_static) con las columnas de df
            const orig = source_static.data;
            let export_data = {};
            // Inicializar export_data con las mismas columnas que orig
            for (const key in orig) {
                export_data[key] = [];
            }
            
            const n = orig['CLUSTER_ID'].length;
            // Filtrar: incluir solo filas cuyo CLUSTER_ID (convertido a string) esté en el conjunto filtrado
            for (let i = 0; i < n; i++){
                if (filtered_clusters.has(String(orig['CLUSTER_ID'][i]))){
                    for (const key in orig){
                        export_data[key].push(orig[key][i]);
                    }
                }
            }
            
            // Convertir export_data a CSV
            const csv = table_to_csv(export_data);
            const filename = 'filtrado.csv';
            
            // Crear Blob y disparar la descarga del CSV
            let blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            if (navigator.msSaveBlob){
                navigator.msSaveBlob(blob, filename);
            } else {
                let link = document.createElement("a");
                link.href = URL.createObjectURL(blob);
                link.download = filename;
                link.target = "_blank";
                link.style.visibility = 'hidden';
                link.dispatchEvent(new MouseEvent('click'));
            }
        """
    )

    export_button.js_on_click(export_callback)

    # Crear layout
    layout = row(column(p, slider, align="center"), column(multi_select, export_button), align="center")

    # Mostrar el gráfico
    show(layout)