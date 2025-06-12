"""
ContornoVibracionesOpt.py

Este script genera diagramas de contorno 2D de vibraciones a cotas específicas:
  1) Lee "pozos_filtrados.csv" (de Visualizacion3d.py).
  2) Reconstruye cargas explosivas (collar y toe).
  3) Lee el polígono original desde "polygon.csv" si existe.
  4) Para cada cota definida, calcula la vibración usando Holmberg-Persson.
  5) Plotea el contorno 2D con pozos y polígono.
  6) Ajusta automáticamente límites para incluir todos los elementos.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap

from Vibrations import holmberg_persson  # Función de vibración de Holmberg-Persson

# Parámetros de explosivo y geometría (ajústalos según tu caso)
RETACADO      =  2.0    # [m] espacio vacío sobre la carga
CHARGE_LENGTH = 10.0    # [m] largo de explosivo dentro del pozo
DIAMETER      = 150     # [m] diámetro del pozo
DENSITY       = 0.8     # [g/cm3] densidad del explosivo
# Constantes del modelo de vibración
CONST_K       = 150     # constante K (ajustar según calibración)
CONST_A       = 0.8     # exponente a (ajustar según calibración)

# Archivos de entrada
tipo_csv    = "pozos_filtrados.csv"
polygon_csv = "polygon.csv"

# Resolución de la grilla XY y margen
PASO_GRID = 0.5      # [m]
MARGEN_REL = 0.1     # Margen relativo (10%)


# Constantes para el cálculo de ppv
young = 70  # [GPa] módulo de Young del terreno
v_p = 2500 # [m/s] velocidad de onda P
r_traccion = 15 # [Mpa] resistencia a la tracción del terreno
ppv_c = (v_p * r_traccion) / (young)  

print(f"PPV_Critico calculado: {ppv_c:.2f} mm/s")



def main():
    # 1) Leer CSV de pozos
    if not os.path.exists(tipo_csv):
        sys.exit(f"ERROR: No encontré '{tipo_csv}'. Ejecuta Visualizacion3d.py primero.")
    df = pd.read_csv(tipo_csv)
    if df.empty:
        sys.exit("ERROR: El CSV de pozos está vacío.")
    for col in ("Coord_X", "Coord_Y"):  # Columnas necesarias
        if col not in df.columns:
            sys.exit(f"ERROR: Falta la columna '{col}' en '{tipo_csv}'.")

    # 2) Reconstruir listas de cargas
    start_z = -RETACADO
    end_z   = -(RETACADO + CHARGE_LENGTH)
    df_sorted = df.copy()
    if "Col" in df_sorted.columns:
        df_sorted = df_sorted.sort_values("Col")
    charges_collar = [(float(r.Coord_X), float(r.Coord_Y), start_z) for _, r in df_sorted.iterrows()]
    charges_toe    = [(float(r.Coord_X), float(r.Coord_Y), end_z)   for _, r in df_sorted.iterrows()]

    # 3) Leer polígono original si existe
    polygon_coords = None
    if os.path.exists(polygon_csv):
        poly_df = pd.read_csv(polygon_csv)
        if {'X','Y'}.issubset(poly_df.columns):
            polygon_coords = poly_df[['X','Y']].values.tolist()
            if polygon_coords[0] != polygon_coords[-1]:
                polygon_coords.append(polygon_coords[0])
        else:
            print(f"Advertencia: '{polygon_csv}' debe contener columnas 'X' y 'Y'.")

    # 4) Determinar límites según pozos y polígono
    xs_p = df["Coord_X"].tolist(); ys_p = df["Coord_Y"].tolist()
    if polygon_coords:
        xs_poly = [p[0] for p in polygon_coords]; ys_poly = [p[1] for p in polygon_coords]
        all_x, all_y = xs_p + xs_poly, ys_p + ys_poly
    else:
        all_x, all_y = xs_p, ys_p
    minx, maxx = min(all_x), max(all_x)
    miny, maxy = min(all_y), max(all_y)
    dx, dy = maxx-minx, maxy-miny
    mx, my = dx*MARGEN_REL, dy*MARGEN_REL
    x_start, x_end = minx-mx, maxx+mx
    y_start, y_end = miny-my, maxy+my


    num_pts = 200     #Si quiero que la cantidad de puntos de energia sea a un paso constante en X e Y 
                      #producira diferentes cantidades de puntos en cada eje y fallará la funcion kleine
           
    #num_pts = int((x_end - x_start) / PASO_GRID) + 1
    x = np.linspace(x_start, x_end , num_pts)
    #num_pts = int((y_end - y_start) / PASO_GRID) + 1
    y = np.linspace(y_start, y_end , num_pts)
    #NX, NY = len(xs), len(ys)

    xx, yy = np.meshgrid(x, y)

    xx = xx.ravel()
    yy = yy.ravel()

    
      # para capturar todo lo que supere 4·perros
    # 6) Definir cotas a evaluar
    #cotas = list(np.arange(0, -18, -3))  # Ajusta según necesites
    
    

    z_cota = 0  # Cota específica a evaluar (puedes cambiarla)
    print(f"→ Procesando cota z = {z_cota:.2f} m")
    
    vibrations = holmberg_persson(xx, yy, z_cota, charges_collar, charges_toe, DIAMETER, DENSITY, CONST_K, CONST_A)
    # Definimos los umbrales:
    thresholds = [
        0,
        0.25 * ppv_c,
        1.0  * ppv_c,
        4.0  * ppv_c,
        vibrations.max()
        ]
    #levels = 10
    # Creamos un colormap discreto de 4 colores:
    cmap = get_cmap("viridis", len(thresholds) - 1)

    # Norm que asocia valores a intervalos:
    norm = BoundaryNorm(thresholds, ncolors=cmap.N, clip=True)

    plt.figure(figsize=(12, 10))
    cs = plt.tricontourf(xx, yy, vibrations,
                        levels=thresholds,
                        cmap=cmap,
                        norm=norm,
                        antialiased=True,
                        extend="max")
    cbar = plt.colorbar(cs, pad=0.02)
    cbar.set_label("Vibraciones (mm/s)")

    # Graficar pozos
    plt.scatter(df["Coord_X"], df["Coord_Y"],
                s=50, facecolors="none",
                edgecolors="black", linewidths=1.5,
                label="Pozos")
    # Graficar polígono si está disponible
    if polygon_coords:
        px, py = zip(*polygon_coords)
        plt.plot(px, py, '-', color='red', linewidth=2, label='Polígono')

    plt.legend(loc="upper right")
    plt.title(f"Diagrama de Contorno de Vibraciones a cota z = {z_cota:.2f} m")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.xlim(x_start, x_end)
    plt.ylim(y_start, y_end)
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    print("\n✅ Diagramas generados correctamente.")
    input("Presiona una tecla para terminar")

if __name__ == "__main__":
    main()