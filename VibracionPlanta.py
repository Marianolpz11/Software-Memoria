# -*- coding: utf-8 -*-
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

from Vibrations import holmberg_persson  # Función de vibración de Holmberg-Persson

# Parámetros de explosivo y geometría (ajústalos según tu caso)
RETACADO      =  2.0    # [m] espacio vacío sobre la carga
CHARGE_LENGTH = 10.0    # [m] largo de explosivo dentro del pozo
DIAMETER      = 0.4     # [m] diámetro del pozo
DENSITY       = 1100.0  # [kg/m³] densidad del explosivo
# Constantes del modelo de vibración
CONST_K       = 1.0     # constante K (ajustar según calibración)
CONST_A       = 0.7     # exponente a (ajustar según calibración)

# Archivos de entrada
tipo_csv    = "pozos_filtrados.csv"
polygon_csv = "polygon.csv"

# Resolución de la grilla XY y margen
PASO_GRID = 0.5      # [m]
MARGEN_REL = 0.1     # Margen relativo (10%)


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

    # 5) Construir grilla XY
    xs = np.arange(x_start, x_end+PASO_GRID, PASO_GRID)
    ys = np.arange(y_start, y_end+PASO_GRID, PASO_GRID)
    NX, NY = len(xs), len(ys)

    # 6) Definir cotas
    cotas = list(np.arange(0, -18, -3))  # ajustar rango según necesidad

    # 7) Calcular y plotear para cada cota
    for z in cotas:
        print(f"→ Procesando vibraciones en z = {z:.2f} m")
        arr2d = np.zeros((NX, NY), dtype=float)
        for i, xv in enumerate(xs):
            for j, yv in enumerate(ys):
                arr2d[i,j] = holmberg_persson(xv, yv, z,
                                              charges_collar,
                                              charges_toe,
                                              DIAMETER,
                                              DENSITY,
                                              CONST_K,
                                              CONST_A)
        # Recorte de valores extremos
        vals = arr2d[~np.isnan(arr2d)];
        if vals.size==0: vmin, vmax = 0,1
        else:
            vmin = np.percentile(vals,5)
            vmax = np.percentile(vals,90)
        levels = np.linspace(vmin, vmax, 40)
        clipped = np.clip(arr2d, vmin, vmax)

        # Plot
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        plt.figure(figsize=(12,10))
        cs = plt.contourf(X, Y, clipped, levels=levels, cmap='viridis', antialiased=True)
        cbar = plt.colorbar(cs, pad=0.02)
        cbar.set_label('Vibración (mm/s) ')
        plt.scatter(df['Coord_X'], df['Coord_Y'], s=50, facecolors='none', edgecolors='black', lw=1.5, label='Pozos')
        if polygon_coords:
            px, py = zip(*polygon_coords)
            plt.plot(px, py, 'r-', lw=2, label='Polígono')
        plt.legend(loc='upper right')
        plt.title(f"Contorno de Vibración a z = {z:.2f} m")
        plt.xlabel('X (m)'); plt.ylabel('Y (m)')
        plt.xlim(x_start, x_end); plt.ylim(y_start, y_end)
        plt.axis('equal'); plt.grid(True, ls='--', alpha=0.5)
        plt.show()

    print("\n✅ Diagramas de vibración generados.")
    input("Presiona una tecla para terminar")

if __name__ == '__main__':
    main()
