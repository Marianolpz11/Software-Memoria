import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Energy import kleine  # Función de energía de Kleine et al.

# Parámetros fijos (ajústalos si lo requieres)
RETACADO      =  2.0    # [m] espacio vacío (sin explosivo) sobre la carga
CHARGE_LENGTH = 10.0    # [m] largo de explosivo dentro del pozo
DIAMETER      = 150     # [g/cm3] diámetro del pozo
DENSITY       = 0.8     # [kg/m³] densidad del explosivo

# Archivos de entrada
tipo_csv    = "pozos_filtrados.csv"
polygon_csv = "polygon.csv"  # Debe contener columnas 'X' y 'Y'

# Paso de discretización de la grilla XY (en metros)
#PASO_GRID = 0.5  # [m]

# Margen relativo para expansión de ejes (10% del rango)
MARGEN_REL = 0.1

def main():
    # 1) Verificar que exista el CSV de pozos
    if not os.path.exists(tipo_csv):
        sys.exit(f"ERROR: No encontré '{tipo_csv}'. Ejecuta Visualizacion3d.py primero.")
    df = pd.read_csv(tipo_csv)
    if df.empty:
        sys.exit("ERROR: El CSV de pozos está vacío.")
    for col in ("Coord_X", "Coord_Y"):
        if col not in df.columns:
            sys.exit(f"ERROR: Falta la columna '{col}' en el CSV de pozos.")

    # 2) Reconstruir las listas de cargas (collar y toe)
    start_z = -RETACADO
    end_z   = -(RETACADO + CHARGE_LENGTH)
    df_sorted = df.copy()
    if "Col" in df_sorted.columns:
        df_sorted = df_sorted.sort_values("Col")
    charges_collar = [
        (float(r.Coord_X), float(r.Coord_Y), start_z)
        for _, r in df_sorted.iterrows()
    ]
    charges_toe = [
        (float(r.Coord_X), float(r.Coord_Y), end_z)
        for _, r in df_sorted.iterrows()
    ]

    # 3) Leer polígono original si existe
    polygon_coords = None
    if os.path.exists(polygon_csv):
        poly_df = pd.read_csv(polygon_csv)
        if {'X', 'Y'}.issubset(poly_df.columns):
            polygon_coords = poly_df[['X', 'Y']].values.tolist()
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

    # 5) Construir vectores XY regulares con resolución constante de 200 puntos
    num_pts = 200
    x = np.linspace(x_start, x_end, num_pts)
    y = np.linspace(y_start, y_end, num_pts)
    xx, yy = np.meshgrid(x, y)
    xx = xx.ravel()
    yy = yy.ravel()

    # 6) Cota específica a evaluar
    z_cota = -7.5  # [m] cambia según lo necesites
    print(f"→ Procesando cota z = {z_cota:.2f} m")

    # 7) Calcular energía y recortar valores extremos
    energy = kleine(xx, yy, z_cota, charges_collar, charges_toe, DIAMETER, DENSITY)
    # Limitar todo valor > 0.7 al nivel máximo de visualización
    display_max = 3.5 # [Kg/m³] nivel máximo de energía a mostrar
    energy = np.clip(energy, 0, display_max)

    # Definir niveles de contorno uniformes de 0 a display_max
    num_levels = 10
    levels = np.linspace(0, display_max, num_levels + 1)

    # 8) Dibujar diagrama de contorno y pozos
    cmap = "jet"
    fig, ax = plt.subplots(figsize=(12, 10))
    cs = ax.tricontourf(
        xx, yy, energy,
        levels=levels,
        cmap=cmap,
        extend="max",       # permite sombrear valores > último nivel
        antialiased=True
    )
    cbar = fig.colorbar(cs, ax=ax, pad=0.02, extend="max")
    cbar.set_label("Energía (Kg/m³)")

    # Pozos (scatter) con tamaño en puntos² constante
    ax.scatter(
        df["Coord_X"], df["Coord_Y"],
        s=50,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Pozos",
        transform=ax.transData
    )

    # Polígono si está disponible
    if polygon_coords:
        px, py = zip(*polygon_coords)
        ax.plot(px, py, '-', color='red', linewidth=2, label='Polígono')

    # 9) Ajustes de ejes y apariencia
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_start, y_end)
    ax.set_title(f"Diagrama de Contorno de Energía a cota z = {z_cota:.2f} m")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    plt.show()
    print("\n✅ Diagramas generados correctamente.")
    input("Presiona una tecla para terminar")

if __name__ == "__main__":
    main()
