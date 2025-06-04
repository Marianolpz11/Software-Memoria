# -*- coding: utf-8 -*-
"""
ContornoSoloCota.py

Este script:
  1) Lee "pozos_filtrados.csv" (generado previamente por Visualizacion3d.py).
  2) Reconstruye las listas de cargas explosivas (collar y toe).
  3) Arma una grilla 3D de energía (llamando a ArrayValues.generate_3d_array
     y a Energy.kleine a través de fill_kleine_channel).
  4) Te pide únicamente la cota (valor Z) para extraer el slice 2D.
  5) Dibuja el diagrama de contorno (contourf) de energía a esa cota,
     superponiendo además cada pozo como un círculo en su posición (X,Y).

La diferencia principal respecto a la versión anterior es que ahora
se ajusta el “L” (lado del cubo 3D) de modo que cubra tanto la extensión
horizontal de los pozos como la profundidad total de cada pozo (desde z=0 hasta
el fondo de la carga), de forma que las cotas abarquen todo el pozo.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# Importa las funciones necesarias
from ArrayValues import generate_3d_array, fill_kleine_channel
# (generate_3d_array devuelve (arr3d, xs, ys, zs);
#  fill_kleine_channel recorre esa grilla y aplica la fórmula de Kleine)

def main():
    # 0) Parámetros fijos (ajústalos si lo requieres)
    # ----------------------------------------------------------------------
    RETACADO      =  2.0    # [m] espacio vacío (sin explosivo) sobre la carga
    CHARGE_LENGTH =  10    # [m] largo de explosivo dentro del pozo
    DIAMETER      =  0.4   # [m] diámetro del pozo
    DENSITY       = 1100.0  # [kg/m³] densidad del explosivo (por ej. ANFO ~ 1600)

    # Nombre del CSV con los pozos filtrados (debe haberlo dejado Visualizacion3d.py)
    CSV_POCES = "pozos_filtrados.csv"

    # PASO de discretización de la grilla 3D (en metros)
    # — mientras más pequeño, mayor detalle y más tiempo de cálculo
    PASO_GRID = 3  # [m]
    # ----------------------------------------------------------------------

    # 1) Verificar que exista el CSV
    if not os.path.exists(CSV_POCES):
        sys.exit(f"ERROR: No encontré '{CSV_POCES}'.\n"
                 "Asegúrate de haber corrido primero Visualizacion3d.py y\n"
                 "que ese CSV esté en el mismo directorio.")

    # 2) Leer el DataFrame con los pozos filtrados
    df = pd.read_csv(CSV_POCES)
    if df.empty:
        sys.exit("ERROR: El CSV está vacío. No hay pozos filtrados para procesar.")

    # Validar que tenga al menos 'Coord_X' y 'Coord_Y'
    for col in ("Coord_X", "Coord_Y"):
        if col not in df.columns:
            sys.exit(f"ERROR: En '{CSV_POCES}' falta la columna '{col}'.")

    # 3) Reconstruir las listas de cargas (collar y toe)
    start_charge_z = -RETACADO
    end_charge_z   = -(RETACADO + CHARGE_LENGTH)

    charges_collar = []
    charges_toe    = []
    # Ordenamos por 'Col' si existe, para mantener el mismo orden que Visualizacion3d.py
    if "Col" in df.columns:
        df_sorted = df.sort_values("Col")
    else:
        df_sorted = df.copy()

    for _, row in df_sorted.iterrows():
        x_i = float(row["Coord_X"])
        y_i = float(row["Coord_Y"])
        # Collar a z = -RETACADO, Toe a z = -(RETACADO + CHARGE_LENGTH)
        charges_collar.append((x_i, y_i, start_charge_z))
        charges_toe.append((x_i, y_i, end_charge_z))

    # 4) Calcular el “centro” horizontal de la grilla
    x0 = df["Coord_X"].mean()
    y0 = df["Coord_Y"].mean()
    # El centro vertical se pone en el tope de la carga (start_charge_z + RETACADO) = 0
    z0 = start_charge_z + RETACADO  # Esto coincide con z = 0 m

    # 5) Determinar L para cubrir:
    #    a) La extensión horizontal de los pozos (mayor_dim),
    #    b) La profundidad total del pozo desde z=0 hasta el fondo de la carga (RETACADO+CHARGE_LENGTH).
    #
    #    Para asegurar que la grilla incluya hasta el fondo de la carga, necesitamos
    #    que L/2 ≥ (RETACADO + CHARGE_LENGTH). Por lo tanto, L_vert = 2*(RETACADO+CHARGE_LENGTH).
    #
    minx, miny = df["Coord_X"].min(), df["Coord_Y"].min()
    maxx, maxy = df["Coord_X"].max(), df["Coord_Y"].max()
    dim_x = maxx - minx
    dim_y = maxy - miny
    mayor_dim = max(dim_x, dim_y)

    # Longitud vertical total (de z=0 hasta fondo de explosivo)
    vertical_total = RETACADO + CHARGE_LENGTH
    L_vert = 2.0 * vertical_total

    # Elegimos el mayor entre la extensión horizontal y la vertical
    L = max(mayor_dim, L_vert)

    # 6) Construir la grilla 3D (cubo de lado L), centrado en (x0, y0, z0)
    print("Construyendo la grilla 3D de energía (puede tardar unos segundos)...")
    arr3d, xs, ys, zs = generate_3d_array((x0, y0, z0), L, PASO_GRID)
    #   - arr3d tiene forma (N, N, N)
    #   - xs, ys, zs son vectores de coordenadas en cada eje.

    # 7) Rellenar arr3d con la energía (fórmula de Kleine) en cada punto
    print("Calculando energía en cada punto de la grilla 3D...")
    fill_kleine_channel(
        arr3d, xs, ys, zs,
        charges_collar, charges_toe,
        DIAMETER, DENSITY
    )
    print("✅ Cálculo de energía 3D completado.\n")

    # 8) Mostrar rango de cotas disponibles y pedir al usuario una cota Z
    z_min = float(zs.min())
    z_max = float(zs.max())
    print(f"Rango de cotas disponibles en la grilla: [{z_min:.2f}  a  {z_max:.2f}]  (m)")
    try:
        cota_input = float(input("→ Ingresa la COTA (valor Z) que quieres evaluar: "))
    except Exception as e:
        sys.exit(f"ERROR leyendo la cota: {e}")

    # 9) Buscar el índice k donde zs[k] está más cerca de cota_input
    idx_k = int(np.argmin(np.abs(zs - cota_input)))
    z_seleccionada = zs[idx_k]
    print(f"Capa seleccionada: índice {idx_k}  →  z = {z_seleccionada:.2f} m\n")

    # 10) Extraer la lámina 2D de energía
    arr2d = arr3d[:, :, idx_k]  # forma (N, N)

    # 11) Generar la malla XY para el contorno
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    # 12) Graficar el diagrama de contorno 2D
    plt.figure(figsize=(10, 8))
    cs = plt.contourf(
        X, Y, arr2d,
        levels=20,              # Número de niveles de contorno (ajusta si quieres más detalle)
        cmap="viridis",
        antialiased=True
    )
    cbar = plt.colorbar(cs, pad=0.02)
    cbar.set_label("Energía (canal 0)")

    # --- SUPERPONER LOS POZOS ---
    # Dibujamos cada pozo como un círculo hueco ('o') en (Coord_X, Coord_Y)
    plt.scatter(
        df["Coord_X"],
        df["Coord_Y"],
        s=40,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        label="Pozos"
    )
    plt.legend(loc="upper right")

    # Etiquetas y estilo
    plt.title(f"Diagrama de Contorno de Energía a cota z = {z_seleccionada:.2f} m")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.show()

    # 13) Guardar la figura si el usuario lo desea
    resp = input("¿Deseas guardar esta figura como PNG? (S/N): ").strip().upper()
    if resp == "S":
        nombre_png = f"contorno_z_{z_seleccionada:.2f}.png".replace(" ", "_").replace(":", "_")
        plt.savefig(nombre_png, dpi=300, bbox_inches="tight")
        print(f"Figura guardada en: {os.path.abspath(nombre_png)}")

    print("\n✅ El diagrama de contorno de energía ha sido generado correctamente.")

if __name__ == "__main__":
    main()
