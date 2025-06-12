
"""
EnergiaLateral.py

Este script:
  1) Lee "pozos_filtrados.csv" (generado previamente por Visualizacion3d.py).
  2) Reconstruye cargas explosivas (collar y toe) a partir del DataFrame.
  3) Lee "polygon.csv" con coordenadas del polígono (opcional).
  4) Solicita corte lateral en ‘X’ o ‘Y’ y distancia desde el origen.
  5) Muestra vista planta con polígono, pozos y línea de corte.
  6) Genera malla lateral con np.meshgrid y calcula energía vectorizada.
  7) Plotea contorno usando plt.tricontourf con arrays 1D (xx, yy, energy).
  8) Superpone líneas de pozos en el slice y aplica márgenes relativos.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Energy import kleine  # Función de energía de Kleine et al.

# Parámetros explosivo/pozo
RETACADO      = 2.0    # [m] espacio vacío sobre la carga
CHARGE_LENGTH = 10.0   # [m] largo de explosivo
DIAMETER      = 150    # [mm] diámetro del pozo
DENSITY       = 0.8    # [g/cm3] densidad explosivo
PASO_GRID     = 0.5    # [m] resolución de grilla
MARGEN_REL    = 0.1    # 10 % margen relativo
LEVELS        = 10     # número de niveles en contour

CSV_POCES   = "pozos_filtrados.csv"
CSV_POLYGON = "polygon.csv"


def main():
    # --- 1) Leer pozos filtrados ---
    if not os.path.exists(CSV_POCES):
        sys.exit(f"ERROR: no encontré '{CSV_POCES}'")
    df = pd.read_csv(CSV_POCES)
    if df.empty:
        sys.exit("ERROR: CSV de pozos vacío.")
    for col in ("Coord_X","Coord_Y"):
        if col not in df.columns:
            sys.exit(f"ERROR: Falta columna '{col}' en CSV de pozos.")

    # --- 2) Reconstruir cargas ---
    z_collar = -RETACADO
    z_toe    = -(RETACADO + CHARGE_LENGTH)
    df_sorted = df.sort_values("Col") if "Col" in df.columns else df
    charges_collar = [(row.Coord_X, row.Coord_Y, z_collar) for _,row in df_sorted.iterrows()]
    charges_toe    = [(row.Coord_X, row.Coord_Y, z_toe)    for _,row in df_sorted.iterrows()]

    # --- 3) Leer polígono (opcional) ---
    if os.path.exists(CSV_POLYGON):
        df_poly = pd.read_csv(CSV_POLYGON)
        if {'X','Y'}.issubset(df_poly.columns):
            poly_coords = df_poly[['X','Y']].values
        else:
            print(f"¡Aviso! '{CSV_POLYGON}' sin columnas X,Y. Omitido.")
            poly_coords = None
    else:
        poly_coords = None
        print(f"¡Aviso! '{CSV_POLYGON}' no encontrado. Omitido.")

    # --- 4) Solicitar corte lateral ---
    x_min,x_max = df.Coord_X.min(), df.Coord_X.max()
    y_min,y_max = df.Coord_Y.min(), df.Coord_Y.max()
    dx = x_max - x_min; dy = y_max - y_min
    eje = input("Corte lateral ('X' o 'Y'): ").strip().upper()
    if eje not in ('X','Y'):
        sys.exit("ERROR: Elige 'X' o 'Y'.")
    maxd = dx if eje=='X' else dy
    print(f"Largo X = {dx:.2f} m, Ancho Y = {dy:.2f} m")
    dist = float(input(f"Distancia 0–{maxd:.2f} m en {eje}: "))
    if not (0<=dist<=maxd):
        sys.exit(f"ERROR: Distancia debe estar en [0,{maxd:.2f}].")
    coord = x_min+dist if eje=='X' else y_min+dist

    # --- 5) Plot vista planta ---
    fig,ax = plt.subplots(figsize=(8,8))
    if poly_coords is not None:
        ax.plot(poly_coords[:,0], poly_coords[:,1],'r-',lw=1.5,label='Polígono')
    ax.scatter(df.Coord_X, df.Coord_Y,s=20,facecolors='none',edgecolors='k',label='Pozos')
    if eje=='X': ax.axvline(coord, color='b', ls='--', label=f'Corte X={coord:.2f}m')
    else:      ax.axhline(coord, color='b', ls='--', label=f'Corte Y={coord:.2f}m')
    ax.set_aspect('equal','box'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('Plan view: pozos y línea de corte'); ax.legend(); ax.grid('--',alpha=0.5)
    plt.tight_layout(); plt.ion(); plt.show()

    # --- 6) Generar malla lateral ---
    if eje=='X':
        horiz = np.arange(y_min, y_max+PASO_GRID, PASO_GRID)
        X = np.full((len(horiz),len(horiz)), coord)  # temp square
    else:
        horiz = np.arange(x_min, x_max+PASO_GRID, PASO_GRID)
        Y = np.full((len(horiz),len(horiz)), coord)
    z_vals = np.arange(z_toe, PASO_GRID, PASO_GRID)
    H_mesh, Z_mesh = np.meshgrid(horiz, z_vals)
    if eje=='X': X_mesh, Y_mesh = np.full_like(H_mesh, coord), H_mesh
    else:        X_mesh, Y_mesh = H_mesh, np.full_like(H_mesh, coord)

    # --- 7) Calcular energía vectorizada ---
    E_mesh = kleine(X_mesh, Y_mesh, Z_mesh, charges_collar, charges_toe, DIAMETER, DENSITY)
    # clip valores extremos
    E_mesh = np.clip(E_mesh, None, 0.7)

    # --- 8) Flatten para tricontourf ---
    xf = H_mesh.ravel(); yf = Z_mesh.ravel(); zf = E_mesh.ravel()

    # --- 9) Dibujar contorno lateral ---
    fig,ax = plt.subplots(figsize=(12,6))
    cs = ax.tricontourf(xf, yf, zf, levels=LEVELS, cmap='jet', antialiased=True)
    cbar = plt.colorbar(cs,pad=0.02,ax=ax); cbar.set_label('Energía (kg/m³)')
    # superponer pozos en slice
    lineas = [ (y if eje=='X' else x) for (x,y,_) in charges_collar if abs((x if eje=='X' else y)-coord)<=PASO_GRID/2 ]
    for off in lineas: ax.plot([off,off],[z_toe,0],'k-',lw=1.2)

    # márgenes relativos
    hmin,hmax = horiz.min(), horiz.max(); zmin,zmax = z_vals.min(),z_vals.max()
    mh = (hmax-hmin)*MARGEN_REL; mz = (zmax-zmin)*MARGEN_REL
    ax.set_xlim(hmin-mh, hmax+mh); ax.set_ylim(zmin-mz, zmax+mz)
    ax.set_aspect('equal','box')
    ax.set_xlabel('Y (m)' if eje=='X' else 'X (m)'); ax.set_ylabel('Z (m)')
    ax.set_title(f'Vista lateral energía → corte {eje}, dist={dist:.2f}m'); ax.grid('--',alpha=0.5)
    plt.tight_layout(); plt.ion(); plt.show()

    input('Presiona Enter para continuar...')

if __name__=='__main__': main()
