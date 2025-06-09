# -*- coding: utf-8 -*-
"""
ContornoSoloCotaOptimizado.py

Este script optimizado:
  1) Lee "pozos_filtrados.csv" (generado previamente por Visualizacion3d.py).
  2) Reconstruye las listas de cargas explosivas (collar y toe).
  3) Lee las coordenadas del polígono original desde "polygon.csv".
  4) Calcula solo las cotas específicas de energía sin generar toda la grilla 3D.
  5) Para cada cota, genera el diagrama de contorno 2D superpuesto con la ubicación de pozos y el polígono.
  6) Ajusta automáticamente los límites del gráfico para incluir todos los pozos con margen extra.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Energy import kleine  # Función de energía de Kleine et al.

# Parámetros fijos (ajústalos si lo requieres)
RETACADO      =  2.0    # [m] espacio vacío (sin explosivo) sobre la carga
CHARGE_LENGTH = 10.0    # [m] largo de explosivo dentro del pozo
DIAMETER      = 0.4     # [m] diámetro del pozo
DENSITY       = 1100.0  # [kg/m³] densidad del explosivo

# Archivos de entrada
tipo_csv    = "pozos_filtrados.csv"
polygon_csv = "polygon.csv"  # Debe contener columnas 'X' y 'Y'

# Paso de discretización de la grilla XY (en metros)
PASO_GRID = 0.5  # [m]

# Margen relativo para expansión de ejes (10% del rango)
MARGEN_REL = 0.1


def main():
    # 1) Verificar que exista el CSV de pozos
    if not os.path.exists(tipo_csv):
        sys.exit(f"ERROR: No encontré '{tipo_csv}'. Ejecuta Visualizacion3d.py primero.")
    df = pd.read_csv(tipo_csv)
    if df.empty:
        sys.exit("ERROR: El CSV de pozos está vacío.")
    for col in ("Coord_X", "Coord_Y"):  # Columnas necesarias
        if col not in df.columns:
            sys.exit(f"ERROR: Falta la columna '{col}' en el CSV de pozos.")

    # 2) Reconstruir las listas de cargas (collar y toe)
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
            # Asegurar cierre del polígono
            if polygon_coords[0] != polygon_coords[-1]:
                polygon_coords.append(polygon_coords[0])
        else:
            print(f"Advertencia: '{polygon_csv}' debe contener columnas 'X' y 'Y'. Se omitirá polígono.")

    # 4) Determinar límites según pozos y aplicar margen
    minx, maxx = df["Coord_X"].min(), df["Coord_X"].max()
    miny, maxy = df["Coord_Y"].min(), df["Coord_Y"].max()
    rango_x = maxx - minx
    rango_y = maxy - miny
    margen_x = rango_x * MARGEN_REL
    margen_y = rango_y * MARGEN_REL
    x_start = minx - margen_x
    x_end   = maxx + margen_x
    y_start = miny - margen_y
    y_end   = maxy + margen_y

    # 5) Construir vectores XY regulares
    xs = np.arange(x_start, x_end + PASO_GRID, PASO_GRID)
    ys = np.arange(y_start, y_end + PASO_GRID, PASO_GRID)
    NX, NY = len(xs), len(ys)

    # 6) Definir cotas a evaluar
    cotas = list(np.arange(0, -18, -3))  # Ajusta según necesites

    # 7) Calcular y graficar para cada cota
    for z_cota in cotas:
        print(f"→ Procesando cota z = {z_cota:.2f} m")
        arr2d = np.zeros((NX, NY), dtype=float)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                arr2d[i, j] = kleine(x, y, z_cota,
                                     charges_collar,
                                     charges_toe,
                                     DIAMETER,
                                     DENSITY)
        # Recortar valores extremos
        datos = arr2d[~np.isnan(arr2d)]
        if datos.size == 0:
            VMIN, VMAX = 0.0, 1.0
        else:
            VMIN = np.percentile(datos, 5)
            VMAX = np.percentile(datos, 90)
        levels = np.linspace(VMIN, VMAX, 40)
        data_clipped = np.clip(arr2d, VMIN, VMAX)

        # Dibujar contorno
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        plt.figure(figsize=(12, 10))
        cs = plt.contourf(X, Y, data_clipped,
                          levels=levels,
                          cmap="viridis",
                          antialiased=True)
        cbar = plt.colorbar(cs, pad=0.02)
        cbar.set_label("Energía (canal 0)")

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
        plt.title(f"Diagrama de Contorno de Energía a cota z = {z_cota:.2f} m")
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
