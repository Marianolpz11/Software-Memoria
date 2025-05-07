import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, Polygon

def generate_rectangular_pattern(burden, spacing, azimuth_degrees, num_rows, num_cols):
    rotation_radians_from_east = np.radians(90 - azimuth_degrees)
    base_coordinates = []

    for i in range(num_rows):
        for j in range(num_cols):
            base_x = j * spacing
            base_y = i * burden
            base_coordinates.append({'Row': i, 'Col': j, 'Base_X': base_x, 'Base_Y': base_y})

    df_base = pd.DataFrame(base_coordinates)

    cos_theta = np.cos(rotation_radians_from_east)
    sin_theta = np.sin(rotation_radians_from_east)
    df_base['Coord_X'] = df_base['Base_X'] * cos_theta - df_base['Base_Y'] * sin_theta
    df_base['Coord_Y'] = df_base['Base_X'] * sin_theta + df_base['Base_Y'] * cos_theta

    df_coords = df_base[['Row', 'Col', 'Coord_X', 'Coord_Y']].copy()
    df_coords['Hole_ID'] = [f'P-{i+1:03d}' for i in df_coords.index]
    df_coords = df_coords[['Hole_ID', 'Row', 'Col', 'Coord_X', 'Coord_Y']]
    return df_coords

def get_polygon_by_click(df_coords):
    print("\nHaz clic para definir el polígono sobre el patrón completo. Pulsa ENTER cuando termines.")
    plt.figure(figsize=(10, 8))
    plt.scatter(df_coords['Coord_X'], df_coords['Coord_Y'], c='blue', label='Pozos')
    plt.title("Haz clic para definir polígono. ENTER para finalizar.")
    plt.xlabel("Este (m)")
    plt.ylabel("Norte (m)")
    plt.grid(True)
    plt.axis('equal')
    points = plt.ginput(n=-1, timeout=0)
    plt.close()

    if len(points) < 3:
        raise ValueError("Se necesitan al menos 3 puntos para un polígono.")
    return Polygon(points)

def get_polygon_from_user_input():
    print("\nIngrese los vértices del polígono irregular (mínimo 3).")
    print("Formato: x,y (ejemplo: 10,20). Escriba 'fin' para terminar.")

    points = []
    while True:
        user_input = input(f"Punto {len(points)+1}: ")
        if user_input.strip().lower() == "fin":
            break
        try:
            x_str, y_str = user_input.split(",")
            x, y = float(x_str.strip()), float(y_str.strip())
            points.append((x, y))
        except ValueError:
            print("Entrada inválida. Use el formato x,y")

    if len(points) < 3:
        raise ValueError("Se necesitan al menos 3 puntos para formar un polígono.")

    return Polygon(points)

def filter_holes_in_polygon(df_coords, polygon):
    inside_mask = df_coords.apply(
        lambda row: polygon.contains(Point(row['Coord_X'], row['Coord_Y'])), axis=1
    )
    return df_coords[inside_mask].copy()

def visualize_pattern(df_coords, burden, spacing, azimuth_degrees, polygon=None):
    if df_coords.empty:
        print("No hay pozos para visualizar.")
        return

    plt.figure(figsize=(10, 8))
    plt.scatter(df_coords['Coord_X'], df_coords['Coord_Y'], marker='o', color='blue', s=50, label='Pozos')

    for index, row in df_coords.iterrows():
        plt.text(row['Coord_X'], row['Coord_Y'], row['Hole_ID'], fontsize=8, ha='left', va='bottom')

    rotation_radians_from_east = np.radians(90 - azimuth_degrees)
    cos_theta = np.cos(rotation_radians_from_east)
    sin_theta = np.sin(rotation_radians_from_east)
    vector_to_face_x = burden * sin_theta
    vector_to_face_y = -burden * cos_theta

    row0_holes = df_coords[df_coords['Row'] == 0]
    if not row0_holes.empty:
        first_hole_row0 = row0_holes.iloc[0]
        last_hole_row0 = row0_holes.iloc[-1]

        free_face_point1_x = first_hole_row0['Coord_X'] + vector_to_face_x
        free_face_point1_y = first_hole_row0['Coord_Y'] + vector_to_face_y
        free_face_point2_x = last_hole_row0['Coord_X'] + vector_to_face_x
        free_face_point2_y = last_hole_row0['Coord_Y'] + vector_to_face_y

        plt.plot(
            [free_face_point1_x, free_face_point2_x],
            [free_face_point1_y, free_face_point2_y],
            color='red',
            linestyle='--',
            linewidth=2,
            label='Cara libre'
        )

    if polygon is not None:
        x_poly, y_poly = polygon.exterior.xy
        plt.plot(x_poly, y_poly, color='green', linestyle='-', linewidth=2, label='Polígono')

    plt.title(f'Vista en planta\nBurden={burden}m, Espaciamiento={spacing}m, Azimuth={azimuth_degrees}°')
    plt.xlabel('Coordenada este (m)')
    plt.ylabel('Coordenada norte (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.axhline(0, color='grey', lw=0.5, linestyle='--')
    plt.axvline(0, color='grey', lw=0.5, linestyle='--')
    plt.legend()
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    print("--- Generando patrón de perforación ---")

    try:
        burden_input = float(input("Ingrese Burden (en metros): "))
        spacing_input = float(input("Ingrese espaciamiento (en metros): "))
        azimuth_input = float(input("Ingrese Azimuth (en grados, 0=Norte, horario): "))
        num_rows_input = int(input("Ingrese número de filas: "))
        num_cols_input = int(input("Ingrese número de columnas: "))
    except ValueError:
        print("Input inválido. Ingrese valores numéricos correctos.")
        exit()

    df_pattern_coords = generate_rectangular_pattern(
        burden_input, spacing_input, azimuth_input, num_rows_input, num_cols_input
    )

    # Selección de método para definir el polígono
    try:
        print("\nSeleccione método para definir el polígono:")
        print(" I - Interactivo (clic en gráfico)")
        print(" C - Coordenadas manuales")
        mode = input("Ingrese opción (I/C): ").strip().upper()

        if mode == "I":
            polygon = get_polygon_by_click(df_pattern_coords)
        elif mode == "C":
            polygon = get_polygon_from_user_input()
        else:
            raise ValueError("Opción inválida. Use 'I' o 'C'.")

        df_filtered = filter_holes_in_polygon(df_pattern_coords, polygon)
        print(f"\nPozos dentro del polígono: {len(df_filtered)} encontrados.")

        if df_filtered.empty:
            print("No se encontraron pozos dentro del polígono. No se generará visualización.")
        else:
            print("\nCoordenadas de los pozos dentro del polígono:")
            print(df_filtered)
            visualize_pattern(df_filtered, burden_input, spacing_input, azimuth_input, polygon)

    except Exception as e:
        print(f"\nError al definir el polígono: {e}")
        exit()

    print("\nProceso completado.")
    print("Gracias por usar el programa.")
    print("Hasta luego.")