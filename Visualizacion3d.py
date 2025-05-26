import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Permite proyección 3D
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, MultiLineString
import sys, os  # Para manejo de sistema y rutas

# --- GENERACIÓN DEL PATRÓN DE PERFORACIÓN (2D) CON EXTENSIÓN DE CARA LIBRE ---
def generate_pattern_from_free_face(free_face_points, burden, spacing, num_rows):
    """
    Genera coordenadas de pozos basados en una polilínea de cara libre
    extendida para cubrir espacios en blanco fuera del polígono.
    """
    if len(free_face_points) < 2:
        print("Error: al menos 2 puntos para definir la cara libre.")
        return pd.DataFrame()

    # Extender la polilínea en ambos extremos para cubrir espacios laterales
    pts = list(free_face_points)
    v0 = np.array(pts[1]) - np.array(pts[0]); d0 = np.linalg.norm(v0)
    if d0 > 0:
        pts.insert(0, tuple(np.array(pts[0]) - (v0/d0)*d0))
    vn = np.array(pts[-1]) - np.array(pts[-2]); dn = np.linalg.norm(vn)
    if dn > 0:
        pts.append(tuple(np.array(pts[-1]) + (vn/dn)*dn))

    free_face_line = LineString(pts)
    all_holes, hole_id = [], 0

    # Generar offsets paralelos (filas)
    for i in range(num_rows):
        offset = (i+1) * burden
        try:
            geom = free_face_line.parallel_offset(offset, 'right', join_style=2)
            lines = list(geom.geoms) if isinstance(geom, MultiLineString) else ([geom] if isinstance(geom, LineString) else [])
        except Exception as e:
            print(f"Advertencia offset fila {i}: {e}")
            continue

        col_idx = 0
        for ln in lines:
            if not isinstance(ln, LineString) or ln.length < 1e-9:
                continue
            distances = np.arange(0, ln.length + 1e-9, spacing)
            for d in distances:
                pt = ln.interpolate(d)
                hole_id += 1
                all_holes.append({
                    'Hole_ID': f'P-{hole_id:03d}',
                    'Row': i,
                    'Col': col_idx,
                    'Coord_X': pt.x,
                    'Coord_Y': pt.y
                })
                col_idx += 1

    if not all_holes:
        print("No se generaron pozos.")
        return pd.DataFrame()
    return pd.DataFrame(all_holes)[['Hole_ID','Row','Col','Coord_X','Coord_Y']]

# --- FILTRADO DE POZOS DENTRO DE UN POLÍGONO ---
def filter_holes_in_polygon(df_coords, polygon):
    """Filtra pozos que quedan dentro del polígono."""
    if polygon is None:
        return pd.DataFrame()
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    pts = df_coords.apply(lambda r: Point(r['Coord_X'], r['Coord_Y']), axis=1)
    mask = pts.apply(polygon.contains)
    return df_coords[mask].copy()

# --- RELLENO DE ESPACIOS EN BLANCO POR FILA ---
def fill_blanks_by_row(df_all, df_filtered):
    """
    Para cada fila, extiende los pozos filtrados para cubrir desde
    el mínimo hasta el máximo índice de columna.
    """
    rows = []
    for r in sorted(df_all['Row'].unique()):
        all_r = df_all[df_all['Row'] == r]
        filt_r = df_filtered[df_filtered['Row'] == r]
        if filt_r.empty:
            continue
        c0, c1 = filt_r['Col'].min(), filt_r['Col'].max()
        rows.append(all_r[(all_r['Col'] >= c0) & (all_r['Col'] <= c1)])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# --- VISUALIZACIÓN 2D ---
def visualize_combined_pattern(df_all, df_filled, burden, spacing, free_face_line=None, polygon=None):
    plt.figure(figsize=(12, 9))
    if free_face_line:
        x, y = free_face_line.xy
        plt.plot(x, y, 'r-', lw=2, label='Cara Libre Extendida')
    if not df_all.empty:
        plt.scatter(df_all['Coord_X'], df_all['Coord_Y'], c='gray', s=20, alpha=0.5, label='Todos Pozos')
    if not df_filled.empty:
        plt.scatter(df_filled['Coord_X'], df_filled['Coord_Y'], c='blue', s=50, label='Pozos Cubiertos')
        for _, g in df_filled.sort_values('Col').groupby('Row'):
            plt.plot(g['Coord_X'], g['Coord_Y'], ':', lw=1, alpha=0.7)
    if polygon and polygon.is_valid:
        x, y = polygon.exterior.xy
        plt.plot(x, y, 'g--', lw=2, label='Polígono')
    plt.title(f'Patrón Planta – Burden={burden}m, Esp={spacing}m')
    plt.xlabel('Este (m)')
    plt.ylabel('Norte (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

# --- VISUALIZACIÓN 3D ---
def visualize_3d_pattern(df_all, df_filled, hole_length, elev=30, azim=45):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    for _, r in df_all.iterrows():
        ax.plot([r['Coord_X']] * 2, [r['Coord_Y']] * 2, [0, -hole_length], c='gray', alpha=0.3)
    for _, r in df_filled.iterrows():
        ax.plot([r['Coord_X']] * 2, [r['Coord_Y']] * 2, [0, -hole_length], c='blue', lw=2)
    ax.set_xlabel('Este (m)')
    ax.set_ylabel('Norte (m)')
    ax.set_zlabel('Profundidad (m)')
    ax.view_init(elev=elev, azim=azim)
    xs, ys = df_all['Coord_X'], df_all['Coord_Y']
    ax.set_xlim(xs.min() - 0.1 * (xs.max() - xs.min()), xs.max() + 0.1 * (xs.max() - xs.min()))
    ax.set_ylim(ys.min() - 0.1 * (ys.max() - ys.min()), ys.max() + 0.1 * (ys.max() - ys.min()))
    ax.set_zlim(-hole_length, 0)
    plt.title(f'3D Isométrico – Largo={hole_length}m')
    plt.show()

# --- DEFINICIÓN DE POLÍGONO (INTERACTIVO/MANUAL) ---
def get_polygon_from_user_input_click(df_coords=None):
    print("--- Definir Polígono (Clic) ---")
    fig, ax = plt.subplots(figsize=(14, 10))
    if df_coords is not None and not df_coords.empty:
        ax.scatter(df_coords['Coord_X'], df_coords['Coord_Y'], c='gray', alpha=0.5)
        mnx, mxx = df_coords['Coord_X'].min(), df_coords['Coord_X'].max()
        mny, mxy = df_coords['Coord_Y'].min(), df_coords['Coord_Y'].max()
        ax.set_xlim(mnx - (mxx - mnx) * 0.1, mxx + (mxx - mnx) * 0.1)
        ax.set_ylim(mny - (mxy - mny) * 0.1, mxy + (mxy - mny) * 0.1)
    else:
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
    ax.set_title('Haz clic para definir polígono (mínimo 3).')
    ax.set_xlabel('Este (m)')
    ax.set_ylabel('Norte (m)')
    ax.grid(True)
    ax.axis('equal')
    pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)
    if len(pts) < 3:
        print("No se definió polígono.")
        return None
    poly = Polygon(pts)
    return poly.buffer(0) if not poly.is_valid else poly

def get_polygon_manual():
    print("--- Definir Polígono (Manual) ---")
    pts = []
    while True:
        s = input(f"Vértice {len(pts)+1} (x,y o 'fin'): ")
        if s.lower() == 'fin':
            break
        try:
            x, y = map(float, s.split(','))
            pts.append((x, y))
        except:
            print("Formato inválido.")
    if len(pts) < 3:
        sys.exit("Se requieren 3 puntos.")
    p = Polygon(pts)
    return p.buffer(0) if not p.is_valid else p

def get_free_face_from_polygon(poly):
    coords = list(poly.exterior.coords)
    nv = len(coords) - 1
    print("Índices disponibles: " + ", ".join(str(i) for i in range(nv)))
    while True:
        seq = input("Índices adyacentes (ej. 0 1 2): ").split()
        if len(seq) < 2:
            continue
        try:
            idx = [int(i) % nv for i in seq]
            pts = [coords[i] for i in idx]
            if LineString(pts).length > 1e-9:
                return pts
        except:
            pass
        print("Secuencia inválida.")

# --- BLOQUE PRINCIPAL ---
if __name__ == "__main__":
    print("--- Generador Patrón 2D/3D ---")
    # Entradas separadas
    try:
        burden_input = float(input("Ingrese Burden (m): "))
        spacing_input = float(input("Ingrese Espaciamiento (m): "))
        num_rows_input = int(input("Ingrese N° de Filas: "))
        hole_length_input = float(input("Ingrese Largo de Pozos (m): "))
        if min(burden_input, spacing_input, num_rows_input, hole_length_input) <= 0:
            raise ValueError
    except:
        sys.exit("Parámetros inválidos.")

    modo = input("Polígono: I-interactivo / C-manual: ").strip().upper()
    main_poly = get_polygon_from_user_input_click(None) if modo == 'I' else get_polygon_manual() if modo == 'C' else None
    if main_poly is None or not main_poly.is_valid:
        sys.exit("Polígono inválido.")

    free_face_pts = get_free_face_from_polygon(main_poly)
    free_face_line = LineString(free_face_pts)

    df_all = generate_pattern_from_free_face(free_face_pts, burden_input, spacing_input, num_rows_input)
    if df_all.empty:
        sys.exit("No se generaron pozos.")

    df_filt = filter_holes_in_polygon(df_all, main_poly)
    df_filled = fill_blanks_by_row(df_all, df_filt)
    print(f"Pozos cubiertos: {len(df_filled)}")

    visualize_combined_pattern(df_all, df_filled, burden_input, spacing_input, free_face_line, main_poly)
    visualize_3d_pattern(df_all, df_filled, hole_length_input)

    if not df_filled.empty:
        csv_fn = "pozos_filtrados.csv"
        df_filled.to_csv(csv_fn, index=False)
        print(f"CSV guardado en: {os.path.abspath(csv_fn)}")
    print("Proceso completado.")
