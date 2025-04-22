# main.py
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon # Shapely para geometría

# --- Estructura para representar los Pozos ---
# Cada fila en este DataFrame representará un pozo individual.
# Considera las columnas necesarias basadas en tu diseño y cálculo:
columnas_pozos = [
    'ID_Pozo',        # Identificador único del pozo (string o int)
    'Coord_X',        # Coordenada X de la boca del pozo (float)
    'Coord_Y',        # Coordenada Y de la boca del pozo (float)
    'Coord_Z',        # Coordenada Z de la boca del pozo (float) - Elevación
    'Profundidad_Total', # Profundidad perforada (float)
    'Diametro',       # Diámetro del pozo (float)
    'Angulo_Buzamiento', # Angulo respecto a la horizontal (grados, float)
    'Angulo_Rumbo',     # Dirección horizontal (grados, float) - Si aplica
    'Tipo_Pozo',      # 'Produccion', 'Contorno', 'Precorte', etc. (string)
    'Estatus',        # 'Diseñado', 'Perforado', 'Cargado', etc. (string)
    # Puedes añadir columnas para resultados del cálculo/simulación:
    'Volumen_Roca_Asociado', # Volumen de roca a fragmentar por este pozo (float)
    'Explosivo_Fondo', # Nombre del explosivo en zona de fondo (string)
    'Longitud_Carga_Fondo', # Longitud de la columna de fondo (float)
    'Explosivo_Columna',# Nombre del explosivo en columna principal (string)
    'Longitud_Carga_Columna',# Longitud de la columna principal (float)
    'Longitud_Taco',    # Longitud del taco (float)
    'Cantidad_Explosivo_Total', # kg o lbs de explosivo total en el pozo (float)
    'Factor_Carga_Resultante', # Factor de carga calculado (float)
    # Resultados de simulación (Ej: contribucion a fragmentacion)
    'Fragmentacion_KuzRam_X50', # Tamaño medio predicho por Kuz-Ram (float)
    # ... otras columnas que necesites
]

# Inicializa un DataFrame vacío para los pozos
df_pozos = pd.DataFrame(columns=columnas_pozos)

# --- Ejemplo: Añadir un pozo de forma manual (para prueba inicial) ---
nuevo_pozo = {
    'ID_Pozo': 'P-001',
    'Coord_X': 100.0,
    'Coord_Y': 200.0,
    'Coord_Z': 50.0,
    'Profundidad_Total': 15.0,
    'Diametro': 0.152, # en metros (ej. 6 pulgadas)
    'Angulo_Buzamiento': 90.0, # Vertical
    'Angulo_Rumbo': 0.0,
    'Tipo_Pozo': 'Produccion',
    'Estatus': 'Diseñado',
    # Deja los resultados vacíos inicialmente
    'Volumen_Roca_Asociado': np.nan,
    'Explosivo_Fondo': np.nan,
    'Longitud_Carga_Fondo': np.nan,
    'Explosivo_Columna': np.nan,
    'Longitud_Carga_Columna': np.nan,
    'Longitud_Taco': np.nan,
    'Cantidad_Explosivo_Total': np.nan,
    'Factor_Carga_Resultante': np.nan,
    'Fragmentacion_KuzRam_X50': np.nan
}

# Añadir el pozo al DataFrame (ignora el index para simplicidad)
df_pozos = pd.concat([df_pozos, pd.DataFrame([nuevo_pozo])], ignore_index=True)

print("DataFrame de Pozos Inicial:")
print(df_pozos)
print("-" * 30)