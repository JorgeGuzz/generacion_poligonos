# Generación de polígonos en minería
### Función principal (genera, grafica y exporta los datos de los polígonos generados)
```
def generar_poligonos(
    ruta_csv: str, 
    ruta_outputs: str, 
    nombre_sector: str, 
    limite_toneladas_formacion: int = 200_000, 
    limite_toneladas_disolucion: int = 100_000, 
    restricciones: List[str] = []
    ) -> None:
```
### Ejemplo de uso:
```
RUTA_CSV = "01.Hermosa_BC_0.1 Flagg.csv"
RUTA_OUTPUTS = "Resultados"
SECTOR = "Hermosa"
LIMITE_TONELADAS_FORMACION = 500_000
LIMITE_TONELADAS_DISOLUCION = 300_000
RESTRICCIONES = ["ARQUEOLOGIA", "C_SAL", "EXPLOTADO", "INFRAESTRUCTURA", "LOMAS", "PILAS", "POLVORIN", "HALLAZGOS"]

generar_poligonos(ruta_csv=RUTA_CSV,
                  ruta_outputs=RUTA_OUTPUTS,
                  nombre_sector=SECTOR,
                  limite_toneladas_formacion=LIMITE_TONELADAS_FORMACION,
                  limite_toneladas_disolucion=LIMITE_TONELADAS_DISOLUCION,
                  restricciones=RESTRICCIONES)
```