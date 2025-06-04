import numpy as np  # Biblioteca para cálculos numéricos y manejo de arreglos
import matplotlib.pyplot as plt  # Biblioteca para gráficos 2D
from mpl_toolkits.mplot3d import Axes3D  # Habilita proyección 3D en matplotlib
import pandas as pd  # Biblioteca para manejo de datos en DataFrames
from shapely.geometry import Point, Polygon, LineString, MultiLineString  # Objetos geométricos para GIS
import sys, os  # Manejo de sistema y rutas de archivo
from ArrayValues import generate_populate  # Función externa para generar matriz de energía


# --- GENERACIÓN DEL PATRÓN DE PERFORACIÓN (2D) CON EXTENSIÓN DE CARA LIBRE ---

def generate_pattern_from_free_face(free_face_points, burden, spacing, num_rows):
    """
    Genera un DataFrame con pozos en filas paralelas a una cara libre extendida.
    """
    # Verifica que haya al menos dos puntos para definir la cara libre
    if len(free_face_points) < 2:
        print("Error: al menos 2 puntos para definir la cara libre.")
        return pd.DataFrame()

    # Extiende la polilínea en ambos extremos para cubrir huecos laterales
    pts = list(free_face_points)
    v0 = np.array(pts[1]) - np.array(pts[0])  # Vector inicial de la cara libre
    d0 = np.linalg.norm(v0)  # Longitud del vector inicial
    if d0 > 0:
        pts.insert(0, tuple(np.array(pts[0]) - (v0 / d0) * d0))  # Punto extendido al inicio
    vn = np.array(pts[-1]) - np.array(pts[-2])  # Vector final de la cara libre
    dn = np.linalg.norm(vn)  # Longitud del vector final
    if dn > 0:
        pts.append(tuple(np.array(pts[-1]) + (vn / dn) * dn))  # Punto extendido al final

    free_face_line = LineString(pts)  # Crea línea de cara libre extendida
    all_holes, hole_id = [], 0  # Lista para almacenar pozos y contador de IDs

    # Itera sobre cada fila desplazada en burden
    for i in range(num_rows):
        offset = (i + 1) * burden  # Distancia de desplazamiento paralelo
        try:
            geom = free_face_line.parallel_offset(offset, 'right', join_style=2)
            # Convierte geometría múltiple a lista de líneas
            if isinstance(geom, MultiLineString):
                lines = list(geom.geoms)
            elif isinstance(geom, LineString):
                lines = [geom]
            else:
                lines = []
        except Exception as e:
            print(f"Advertencia offset fila {i}: {e}")
            continue

        col_idx = 0  # Índice de columna para cada fila
        for ln in lines:
            if not isinstance(ln, LineString) or ln.length < 1e-9:
                continue  # Descarta líneas inválidas o muy pequeñas
            distances = np.arange(0, ln.length + 1e-9, spacing)  # Puntos equidistantes
            for d in distances:
                pt = ln.interpolate(d)  # Punto interpolado sobre la línea
                hole_id += 1
                all_holes.append({
                    'Hole_ID': f'P-{hole_id:03d}',
                    'Row':     i,
                    'Col':     col_idx,
                    'Coord_X': pt.x,
                    'Coord_Y': pt.y
                })
                col_idx += 1  # Incrementa columna

    # Devuelve DataFrame con columnas ordenadas si hay pozos
    if not all_holes:
        print("No se generaron pozos.")
        return pd.DataFrame()

    return pd.DataFrame(all_holes)[['Hole_ID','Row','Col','Coord_X','Coord_Y']]


# --- FILTRADO DE POZOS DENTRO DE UN POLÍGONO ---

def filter_holes_in_polygon(df_coords, polygon):
    """
    Retiene solo pozos cuyos puntos caen dentro del polígono dado.
    """
    if polygon is None:
        return pd.DataFrame()
    if not polygon.is_valid:
        polygon = polygon.buffer(0)  # Corrige geometría si no es válida
    # Crea serie de objetos Point y aplica contains
    pts = df_coords.apply(lambda r: Point(r['Coord_X'], r['Coord_Y']), axis=1)
    mask = pts.apply(polygon.contains)
    return df_coords[mask].copy()  # Devuelve copia de filas dentro


# --- RELLENO DE ESPACIOS EN BLANCO POR FILA ---

def fill_blanks_by_row(df_all, df_filtered):
    """
    Extiende pozos filtrados en cada fila hasta cubrir min y max índice de columna.
    """
    rows = []
    for r in sorted(df_all['Row'].unique()):  # Itera cada fila existente
        all_r  = df_all[df_all['Row'] == r]  # Todos pozos de la fila
        filt_r = df_filtered[df_filtered['Row'] == r]  # Pozos filtrados en fila
        if filt_r.empty:
            continue  # Omite filas sin pozos filtrados
        c0, c1 = filt_r['Col'].min(), filt_r['Col'].max()  # Rangos de columnas
        # Añade pozos entre los índices mínimo y máximo
        rows.append(all_r[(all_r['Col'] >= c0) & (all_r['Col'] <= c1)])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --- VISUALIZACIÓN 2D (SOLO POZOS DENTRO DEL POLÍGONO) ---

def visualize_combined_pattern(df_all, df_filled,
                               burden, spacing,
                               free_face_line=None, polygon=None):
    """
    Grafica planta mostrando cara libre, pozos dentro, cubiertos y polígono.
    """
    # Filtra pozos por polígono si existe
    if polygon is not None:
        df_all    = filter_holes_in_polygon(df_all, polygon)
        df_filled = filter_holes_in_polygon(df_filled, polygon)

    plt.figure(figsize=(12, 9))  # Crea figura 2D
    if free_face_line:
        x, y = free_face_line.xy
        plt.plot(x, y, 'r-', lw=2, label='Cara Libre Extendida')

    # Dibuja pozos dentro
    if not df_all.empty:
        plt.scatter(df_all['Coord_X'], df_all['Coord_Y'],
                    c='gray', s=20, alpha=0.5, label='Pozos dentro')

    # Dibuja pozos cubiertos y líneas de fila
    if not df_filled.empty:
        plt.scatter(df_filled['Coord_X'], df_filled['Coord_Y'],
                    c='blue', s=50, label='Pozos cubiertos')
        for _, g in df_filled.sort_values('Col').groupby('Row'):
            plt.plot(g['Coord_X'], g['Coord_Y'], ':', lw=1, alpha=0.7)

    # Dibuja polígono principal
    if polygon is not None and polygon.is_valid:
        x, y = polygon.exterior.xy
        plt.plot(x, y, 'g--', lw=2, label='Polígono')

    # Configuración de ejes, título y leyenda
    plt.title(f'Patrón Planta – Burden={burden}m, Esp={spacing}m')
    plt.xlabel('Este (m)'); plt.ylabel('Norte (m)')
    plt.grid(True); plt.axis('equal'); plt.legend()
    plt.show()


# --- VISUALIZACIÓN 3D (SOLO POZOS DENTRO DEL POLÍGONO) ---

def visualize_3d_pattern(df_all, df_filled,
                         hole_length, charge_length,
                         retacado, pasadura, ax,
                         elev=30, azim=45,
                         polygon=None):
    """
    Grafica en 3D pozos, retacado, carga y pasadura con ejes igualados.
    """
    # Filtra pozos por polígono si existe
    if polygon is not None:
        df_all    = filter_holes_in_polygon(df_all, polygon)
        df_filled = filter_holes_in_polygon(df_filled, polygon)

    # Dibuja cada pozo completo en gris
    for _, r in df_all.iterrows():
        ax.plot([r['Coord_X']]*2, [r['Coord_Y']]*2,
                [0, -hole_length],
                c='gray', alpha=0.3,
                label='Pozos dentro' if _==0 else "")

    # Dibuja retacado, carga explosiva y pasadura
    first_ret = True; first_charge = True; first_pas = True
    for _, r in df_filled.iterrows():
        x0, y0 = r['Coord_X'], r['Coord_Y']
        # Retacado (amarillo)
        ax.plot([x0, x0], [y0, y0], [0, -retacado],
                c='yellow', lw=3,
                label='Retacado' if first_ret else "")
        first_ret = False

        # Carga explosiva (rojo)
        start_charge = -retacado
        end_charge   = -(retacado + charge_length)
        ax.plot([x0, x0], [y0, y0],
                [start_charge, end_charge],
                c='red', lw=3,
                label='Carga explosiva' if first_charge else "")
        first_charge = False

        # Pasadura (verde)
        top_pasadura = hole_length - pasadura
        ax.plot([x0, x0], [y0, y0],
                [-top_pasadura, -hole_length],
                c='green', lw=3,
                label='Pasadura' if first_pas else "")
        first_pas = False

    # Configura ejes y vista
    ax.set_xlabel('Este (m)'); ax.set_ylabel('Norte (m)'); ax.set_zlabel('Profundidad (m)')
    ax.view_init(elev=elev, azim=azim)

    # Igualar escalas en X, Y y Z
    xs, ys = df_all['Coord_X'], df_all['Coord_Y']
    zs = np.array([0, -hole_length])
    max_range = max(xs.max()-xs.min(), ys.max()-ys.min(), abs(zs.max()-zs.min()))
    mid_x = (xs.max()+xs.min())/2; mid_y = (ys.max()+ys.min())/2; mid_z = (zs.max()+zs.min())/2
    ax.set_xlim(mid_x-max_range/2, mid_x+max_range/2)
    ax.set_ylim(mid_y-max_range/2, mid_y+max_range/2)
    ax.set_zlim(mid_z-max_range/2, mid_z+max_range/2)

    ax.legend(loc='upper right')  # Agrega leyenda
    plt.title(f'3D Isométrico – Pozo={hole_length}m, Carga={charge_length}m, Retacado={retacado}m, Pasadura={pasadura}m')
    plt.ion(); plt.show()


# --- DEFINICIÓN DE POLÍGONO POR USUARIO (INTERACTIVO/MANUAL) ---

def get_polygon_from_user_input_click(df_coords=None):
    print("--- Definir Polígono (Clic) ---")
    fig, ax = plt.subplots(figsize=(14, 10))  # Prepara figura para clics
    if df_coords is not None and not df_coords.empty:
        ax.scatter(df_coords['Coord_X'], df_coords['Coord_Y'], c='gray', alpha=0.5)
        mnx, mxx = df_coords['Coord_X'].min(), df_coords['Coord_X'].max()
        mny, mxy = df_coords['Coord_Y'].min(), df_coords['Coord_Y'].max()
        ax.set_xlim(mnx-0.1*(mxx-mnx), mxx+0.1*(mxx-mnx))
        ax.set_ylim(mny-0.1*(mxy-mny), mxy+0.1*(mxy-mny))
    else:
        ax.set_xlim(-50, 50); ax.set_ylim(-50, 50)

    ax.set_title('Haz clic para definir polígono (mínimo 3).')
    ax.set_xlabel('Este (m)'); ax.set_ylabel('Norte (m)')
    ax.grid(); ax.axis('equal')
    pts = plt.ginput(n=-1, timeout=0, show_clicks=True)  # Captura clicks
    plt.close(fig)
    if len(pts) < 3:
        print("No se definió polígono."); return None
    poly = Polygon(pts)
    return poly.buffer(0) if not poly.is_valid else poly


def get_polygon_manual():
    """
    Permite al usuario ingresar manualmente los vértices de un polígono en consola.
    Retorna un objeto Polygon válido, o termina el programa si hay menos de 3 puntos.
    """
    print("--- Definir Polígono (Manual) ---")  # Mensaje inicial para el usuario
    pts = []  # Lista para almacenar los vértices ingresados

    while True:
        # Solicita coordenada o 'fin' para terminar
        s = input(f"Vértice {len(pts)+1} (x,y o 'fin'): ")

        if s.lower() == 'fin':
            break  # Sale del bucle cuando el usuario termina la entrada
        try:
            # Convierte la cadena "x,y" en dos floats x e y
            x, y = map(float, s.split(','))
            pts.append((x, y))  # Añade el punto a la lista de vértices
        except:
            print("Formato inválido. Use 'x,y' con números flotantes.")  # Notifica error de formato

    # Verifica que haya al menos tres vértices para formar un polígono
    if len(pts) < 3:
        sys.exit("Se requieren al menos 3 puntos para definir un polígono.")

    # Crea el polígono a partir de los vértices ingresados
    p = Polygon(pts)
    # Si la geometría no es válida, corrige con buffer(0)
    return p.buffer(0) if not p.is_valid else p



def get_free_face_from_polygon(poly):
    """
    Permite al usuario seleccionar una secuencia de vértices adyacentes del polígono
    para definir la cara libre. Retorna la lista de puntos seleccionados.
    """
    # Obtiene lista de coordenadas del exterior del polígono
    coords = list(poly.exterior.coords)
    # El último vértice es igual al primero, por eso nv = len(coords) - 1
    nv = len(coords) - 1
    # Muestra al usuario los índices válidos
    print("Índices disponibles:", ", ".join(str(i) for i in range(nv)))

    while True:
        # Pide al usuario ingresar índices separados por espacios
        seq = input("Índices adyacentes (ej. 0 1 2): ").split()
        if len(seq) < 2:
            continue  # Necesita al menos dos índices para formar una cara
        try:
            # Convierte cada índice a entero y aplica módulo para manejo circular
            idx = [int(i) % nv for i in seq]
            # Extrae las coordenadas correspondientes
            pts = [coords[i] for i in idx]
            # Verifica que la línea definida tenga longitud significativa
            if LineString(pts).length > 1e-9:
                return pts  # Retorna los puntos válidos
        except:
            # Ignora errores de conversión y continúa solicitando
            pass
        print("Secuencia inválida. Asegúrese de ingresar índices adyacentes válidos.")  # Indica error y repite


# --- BLOQUE PRINCIPAL ---

if __name__ == "__main__":
    print("--- Generador Patrón 2D/3D con Carga, Retacado y Pasadura ---")
    modo = input("Modo manual: M o Test: T ").strip().upper()  # Selección de modo
    if modo == 'M':
        try:
            burden_input      = float(input("Ingrese Burden (m): "))
            spacing_input     = float(input("Ingrese Espaciamiento (m): "))
            num_rows_input    = int(input("Ingrese N° de Filas: "))
            hole_length_input = float(input("Ingrese Largo de Pozos (m): "))
            charge_length     = float(input("Ingrese Largo de Carga Explosiva (m): "))
            retacado          = float(input("Ingrese Largo de Retacado (m): "))
            pasadura          = float(input("Ingrese Largo de Pasadura (m): "))
            diameter          = float(input("Ingrese Diámetro de Pozos (m): "))
            density           = float(input("Ingrese Densidad de Explosivo (kg/m³): "))
            # Verifica coherencia de parámetros y suma de longitudes
            if (burden_input <= 0 or spacing_input <= 0 or num_rows_input <= 0 or
                hole_length_input <= 0 or charge_length < 0 or retacado < 0 or pasadura < 0 or
                retacado + charge_length + pasadura > hole_length_input):
                raise ValueError
        except:
            sys.exit("Parámetros inválidos o suman más que el largo del pozo.")
    elif modo == 'T':
        # Parámetros de prueba predefinidos
        burden_input      = 5.0
        spacing_input     = 5.0
        num_rows_input    = 15
        hole_length_input = 15.0
        charge_length     = 10.0
        retacado          = 3.0
        pasadura          = 2.0
        diameter          = 0.4
        density           = 1100.0

    modo = input("Polígono: I-interactivo / C-manual: ").strip().upper()
    if modo == 'I':
        main_poly = get_polygon_from_user_input_click(None)
    elif modo == 'C':
        main_poly = get_polygon_manual()
    else:
        main_poly = None

    if main_poly is None or not main_poly.is_valid:
        sys.exit("Polígono inválido.")

    free_face_pts  = get_free_face_from_polygon(main_poly)  # Define cara libre
    free_face_line = LineString(free_face_pts)  # Línea de cara libre para visualización

    # Genera patrón 2D y filtra pozos
    df_all    = generate_pattern_from_free_face(free_face_pts, burden_input, spacing_input, num_rows_input)
    if df_all.empty:
        sys.exit("No se generaron pozos.")

    df_filt   = filter_holes_in_polygon(df_all, main_poly)
    df_filled = fill_blanks_by_row(df_all, df_filt)
    print(f"Pozos cubiertos: {len(df_filled)}")

    # Visualizaciones 2D y 3D
    visualize_combined_pattern(df_all, df_filled, burden_input, spacing_input, free_face_line, main_poly)
    fig = plt.figure(figsize=(12, 9)); ax = fig.add_subplot(111, projection='3d')
    visualize_3d_pattern(df_all, df_filled, hole_length_input, charge_length, retacado, pasadura, ax, elev=30, azim=45, polygon=main_poly)

    modo = input("A: Generar matriz de energia traslucida y B opaca. ").strip().upper()
    # Genera matriz de energía usando función externa
    if modo == 'A':
        generate_populate(ax, df_filled, retacado, charge_length, diameter, density, main_poly, opacity=True)
    elif modo == 'B':
        generate_populate(ax, df_filled, retacado, charge_length, diameter, density, main_poly, opacity=False)

    input("Presiona una tecla para terminar")

    # Guarda CSV con pozos filtrados si existen
    if not df_filled.empty:
        csv_fn = "pozos_filtrados.csv"
        df_filled.to_csv(csv_fn, index=False)
        print(f"CSV guardado en: {os.path.abspath(csv_fn)}")
    print("Proceso completado.")
