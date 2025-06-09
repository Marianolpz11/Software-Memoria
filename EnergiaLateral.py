# -*- coding: utf-8 -*-
"""
VisualizacionLateral.py

1) Muestra en planta el polígono, los pozos y la línea de corte
2) Traza el slice lateral de energía (función kleine) en el corte elegido
   con vmin = percentil 0 y vmax = percentil 95 del slice
   - Z con 0 m arriba y –profundidad abajo (sin invertir eje)
   - +5 m de margen horizontal
   - Selección de corte por distancia desde el inicio (0 → largo malla)
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Energy import kleine   # tu implementación de la función de energía

# — Parámetros de explosivo / pozo (ajusta si lo necesitas) —
RETACADO      =  2.0    # m  espacio vacío sobre la carga
CHARGE_LENGTH = 10.0   # m  largo del explosivo
DIAMETER      = 0.4    # m  diámetro del pozo
DENSITY       = 1100.0 # kg/m³ densidad del explosivo
PASO_GRID     = 0.5    # m  resolución de la grilla en el corte

CSV_POCES   = "pozos_filtrados.csv"
CSV_POLYGON = "polygon.csv"

# --- 1) Leo CSV de pozos filtrados ---
if not os.path.exists(CSV_POCES):
    sys.exit(f"ERROR: no encontré '{CSV_POCES}'. Ejecuta primero Visualizacion3d.py.")
df_pozos = pd.read_csv(CSV_POCES)

# --- 2) Leo CSV de polígono (si existe) ---
if os.path.exists(CSV_POLYGON):
    df_poly = pd.read_csv(CSV_POLYGON)
    poly_x, poly_y = df_poly["X"].values, df_poly["Y"].values
else:
    df_poly = None
    print(f"¡Aviso! No encontré '{CSV_POLYGON}'. Omito el polígono en el plan view.")

# --- 3) Determino dimensiones de la malla ---
x_min, x_max = df_pozos["Coord_X"].min(), df_pozos["Coord_X"].max()
y_min, y_max = df_pozos["Coord_Y"].min(), df_pozos["Coord_Y"].max()
largo_x = x_max - x_min
largo_y = y_max - y_min

# --- 4) Pido plano de corte y distancia ---
eje = input("Corte en plano ('X' o 'Y'): ").strip().upper()
if eje not in ("X", "Y"):
    sys.exit("ERROR: debes elegir 'X' o 'Y'.")

max_dist = largo_x if eje == "X" else largo_y
print(f"Largo X = {largo_x:.2f} m, Ancho Y = {largo_y:.2f} m")
dist = float(input(f"Ingresa distancia desde 0 hasta {max_dist:.2f} m en {eje}: "))
if dist < 0 or dist > max_dist:
    sys.exit(f"ERROR: la distancia debe estar entre 0 y {max_dist:.2f} m.")

# Traduzco distancia en coordenada real
coord = (x_min + dist) if eje == "X" else (y_min + dist)

# --- 5) Plan view: polígono, pozos y línea de corte ---
plt.figure(figsize=(8, 8))
ax1 = plt.gca()

# Polígono original
if df_poly is not None:
    ax1.plot(poly_x, poly_y, 'r-', lw=1.5, label="Polígono")

# Pozos
ax1.scatter(df_pozos["Coord_X"],
            df_pozos["Coord_Y"],
            s=20, facecolors='none', edgecolors='k',
            label="Pozos")

# Línea de corte
if eje == "X":
    ax1.axvline(x=coord, color="b", ls="--", label=f"Corte X = {coord:.2f} m")
else:
    ax1.axhline(y=coord, color="b", ls="--", label=f"Corte Y = {coord:.2f} m")

ax1.set_aspect('equal', 'box')
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_title("Plan view: polígono, pozos y línea de corte")
ax1.legend(loc="upper right")
ax1.grid(True, ls="--", alpha=0.5)
plt.tight_layout()
plt.ion()
plt.show()

# --- 6) Reconstruyo las cargas (collar y toe) ---
z0 = -RETACADO
z1 = -(RETACADO + CHARGE_LENGTH)
charges_collar = []
charges_toe    = []
df_iter = df_pozos.sort_values("Col") if "Col" in df_pozos.columns else df_pozos
for _, r in df_iter.iterrows():
    xi, yi = float(r["Coord_X"]), float(r["Coord_Y"])
    charges_collar.append((xi, yi, z0))
    charges_toe.   append((xi, yi, z1))

# --- 7) Defino vectores horizontales y verticales para el slice ---
if eje == "X":
    horiz = np.arange(y_min, y_max + PASO_GRID, PASO_GRID)
else:
    horiz = np.arange(x_min, x_max + PASO_GRID, PASO_GRID)
z_vals = np.arange(z1, 0 + PASO_GRID, PASO_GRID)

# --- 8) Calculo la matriz de energía E[z, h] ---
E = np.zeros((len(z_vals), len(horiz)))
for iz, z in enumerate(z_vals):
    for ih, h in enumerate(horiz):
        x = coord if eje == "X" else h
        y = h     if eje == "X" else coord
        E[iz, ih] = kleine(x, y, z,
                           charges_collar, charges_toe,
                           DIAMETER, DENSITY)

# --- 9) Estadísticos: percentil 0 y 95 para escala de colores ---
flat  = E.flatten()
valid = flat[~np.isnan(flat)]
vmin  = np.percentile(valid, 0)
vmax  = np.percentile(valid, 95)

# --- 10) Plot lateral de energía ---
plt.figure(figsize=(12, 6))
cs = plt.contourf(horiz, z_vals, E,
                  levels=60, vmin=vmin, vmax=vmax)
cbar = plt.colorbar(cs, pad=0.02)
cbar.set_label("Energía (kg/m³)")

# Superpongo las líneas de los pozos en el slice
if eje == "X":
    pozos_slice = [y for (x, y, _) in charges_collar
                   if abs(x - coord) <= PASO_GRID/2]
    xlabel = "Y (m)"
else:
    pozos_slice = [x for (x, y, _) in charges_collar
                   if abs(y - coord) <= PASO_GRID/2]
    xlabel = "X (m)"

for off in pozos_slice:
    plt.plot([off, off], [z1, 0], 'k-', lw=1.2)

# Márgenes horizontales (+5 m más)
margen = 5.0
plt.xlim(horiz.min() - margen, horiz.max() + margen)

plt.title(f"Visualización lateral → corte {eje}, dist = {dist:.2f} m")
plt.xlabel(xlabel)
plt.ylabel("Z (m)")
plt.grid(True, ls="--", alpha=0.5)
plt.tight_layout()
plt.ion()
plt.show()

input("Presiona Enter para continuar...")
