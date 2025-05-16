# pattern_from_polygon_polyline_face_v7.py - Add CSV Output

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, MultiLineString
import sys
import os # Importar el módulo os para manejar rutas de archivo

def generate_pattern_from_free_face(free_face_points, burden, spacing, num_rows):
    """
    Genera coordenadas para una malla de perforación basada en una línea de cara libre (polilínea).

    La malla se genera creando líneas paralelas a la cara libre (polilínea), separadas por 'burden',
    y colocando pozos sobre estas líneas, separados por 'spacing'. Los pozos se colocan
    interpolando a lo largo de la longitud total de cada línea de fila para asegurar
    un espaciamiento uniforme.

    Args:
        free_face_points (list of tuple): Lista de tuplas (x, y) que definen la polilínea de la cara libre.
                                         Esta polilínea puede tener más de dos puntos y debe estar
                                         compuesta por vértices consecutivos del polígono principal.
                                         Los puntos deben estar en un orden consistente
                                         (ej. de inicio a fin de la polilínea, tal que el macizo
                                         quede a la derecha).
        burden (float): Distancia entre filas (perpendicular a la cara libre) en metros.
        spacing (float): Distancia entre pozos en una fila (a lo largo de la cara libre) en metros.
        num_rows (int): Número de filas a generar desde la cara libre hacia adentro.

    Returns:
        pandas.DataFrame: DataFrame con columnas 'Hole_ID', 'Row', 'Col', 'Coord_X', 'Coord_Y'.
                          'Row' indica la fila desde la cara libre (0 es la primera fila, a 1 Burden).
                          'Col' indica la posición a lo largo de la fila (aproximada).
                          Retorna un DataFrame vacío si hay un error o no se generan pozos.
    """
    if len(free_face_points) < 2:
        print("Error: Se necesitan al menos 2 puntos para definir la línea de la cara libre.")
        return pd.DataFrame() # Retorna DataFrame vacío en caso de error

    # La cara libre ahora es una polilínea, LineString maneja esto correctamente
    free_face_line = LineString(free_face_points)
    all_holes = []
    hole_id_counter = 0

    # Determinar el lado del offset. Asumimos que 'right' es hacia el macizo
    # si los puntos de la cara libre se definen en un orden determinado (ej. borde izquierdo al borde derecho del banco).
    # Si los pozos se generan en la dirección incorrecta, invertir el orden de los puntos de entrada de la cara libre.
    offset_side = 'right'

    for i in range(num_rows): # i = row index (0 para la primera fila generada, que estará a 1*Burden)
        # Inicializar lines_to_process como una lista vacía al comienzo de CADA iteración
        lines_to_process = []

        # Calcular la línea offset para la fila actual
        # La primera fila es a 1*burden, la segunda a 2*burden, etc.
        offset_distance = (i + 1) * burden

        # Generar línea paralela offset. Usar 2D only.
        try:
             # Usar join_style=2 (round) o 1 (mitre) o 3 (bevel) para manejar las esquinas.
             # mitre_limit es para join_style=1
             # parallel_offset de una LineString puede resultar en otra LineString o un MultiLineString
             current_row_geometry = free_face_line.parallel_offset(offset_distance, offset_side, join_style=2)

             # parallel_offset puede retornar un MultiLineString o LineString
             if isinstance(current_row_geometry, MultiLineString):
                 lines_to_process = list(current_row_geometry.geoms)
             # Si current_row_geometry no es MultiLineString, debe ser una LineString o un objeto vacío.
             elif isinstance(current_row_geometry, LineString):
                 lines_to_process = [current_row_geometry]
             # Si no es ninguno de los anteriores (ej. un punto si la línea se colapsa), lines_to_process permanece como lista vacía.

        except Exception as e:
             print(f"Advertencia: No se pudo crear la línea offset para la fila {i} a distancia {offset_distance}m. Error: {e}")
             print("Esto podría ocurrir con formas de cara libre complejas o Burden grandes.")
             # lines_to_process ya es una lista vacía por la inicialización, no hacemos nada más aquí.
             continue # Saltar el resto del bucle de esta fila


        # Colocar puntos a lo largo de las líneas de la fila actual espaciados por 'spacing'
        # Este bucle ahora es seguro porque lines_to_process siempre es una lista, aunque esté vacía.
        row_holes = []
        current_row_col = 0

        for line_geometry in lines_to_process: # line_geometry es una LineString o parte de una MultiLineString
            # Asegurarse de que es una LineString válida antes de interpolar
            if not isinstance(line_geometry, LineString) or line_geometry.length <= 1e-9 or spacing <= 1e-9:
                 continue # Saltar si no es LineString, no tiene longitud significativa o espaciamiento es cero

            line_length = line_geometry.length

            # Generar puntos a lo largo de esta LineString completa por distancia interpolada
            # Usar np.arange para obtener las distancias a lo largo de la línea donde colocar los puntos.
            # El + 1e-9 asegura que si la longitud es exactamente un múltiplo de spacing, el punto final se incluya.
            distances_to_interpolate = np.arange(0, line_length + 1e-9, spacing)

            for distance_along_line in distances_to_interpolate:
                # Interpolate(distance) devuelve un Point a esa distancia a lo largo de la LineString
                point = line_geometry.interpolate(distance_along_line)

                # Añadir el punto
                hole_id_counter += 1
                row_holes.append({
                    'Hole_ID': f'P-{hole_id_counter:03d}',
                    'Row': i,
                    'Col': current_row_col, # Índice de columna consecutivo a lo largo de la línea(s) de esta fila
                    'Coord_X': point.x,
                    'Coord_Y': point.y
                })
                current_row_col += 1

            # Con np.arange y interpolate(distance), no necesitamos añadir el último punto por separado,
            # ya que np.arange con la tolerancia maneja eso.


        all_holes.extend(row_holes)


    # Crear DataFrame
    if not all_holes:
         print("No se generaron pozos. Verifique los inputs (Burden, Spacing, Num Rows) y la cara libre.")
         return pd.DataFrame()

    df_coords = pd.DataFrame(all_holes)
    # El ID ya se asignó en el bucle
    df_coords = df_coords[['Hole_ID', 'Row', 'Col', 'Coord_X', 'Coord_Y']] # Reordenar columnas


    return df_coords

# --- FUNCIÓN DE VISUALIZACIÓN ---
def visualize_combined_pattern(df_generated_coords, df_filtered_coords, burden, spacing, free_face_line=None, polygon=None):
    """
    Genera un gráfico combinado mostrando la cara libre, todos los pozos generados
    y los pozos filtrados dentro de un polígono.

    Args:
        df_generated_coords (pandas.DataFrame): DataFrame con TODOS los pozos generados.
        df_filtered_coords (pandas.DataFrame): DataFrame con los pozos filtrados (dentro del polígono).
        burden (float): Distancia de Burden (para el título).
        spacing (float): Distancia de Espaciamiento (para el título).
        free_face_line (LineString, optional): Objeto LineString de Shapely que representa la cara libre.
                                              Se dibujará en el gráfico si se proporciona.
        polygon (shapely.geometry.Polygon, optional): Objeto Polígono de Shapely. Se dibujará si se proporciona.
    """
    plt.figure(figsize=(10, 8))

    # Dibujar la línea de la cara libre si se proporciona
    if free_face_line is not None:
        x_ff, y_ff = free_face_line.xy
        plt.plot(x_ff, y_ff, color='red', linestyle='-', linewidth=2, label='Cara Libre (Polilínea Seleccionada)') # Color rojo para destacar la polilínea

    # Dibujar TODOS los pozos generados (fondo, en gris)
    if not df_generated_coords.empty:
        plt.scatter(df_generated_coords['Coord_X'], df_generated_coords['Coord_Y'],
                    marker='o', color='gray', s=20, alpha=0.5, label='Pozos Generados (Todos)')

    # Dibujar los pozos filtrados (primer plano, en azul)
    if not df_filtered_coords.empty:
        plt.scatter(df_filtered_coords['Coord_X'], df_filtered_coords['Coord_Y'],
                    marker='o', color='blue', s=50, label='Pozos Filtrados')

        # Etiquetar pozos filtrados (si no son demasiados)
        if len(df_filtered_coords) < 100: # Umbral para etiquetar
             for index, row in df_filtered_coords.iterrows():
                  plt.text(row['Coord_X'], row['Coord_Y'], row['Hole_ID'], fontsize=8, ha='left', va='bottom')
        else:
             print(f"Saltando etiquetado de pozos filtrados. Demasiados pozos ({len(df_filtered_coords)}).")


    # Dibujar el polígono si se proporciona
    if polygon is not None and polygon.is_valid:
        x_poly, y_poly = polygon.exterior.xy
        plt.plot(x_poly, y_poly, color='green', linestyle='--', linewidth=2, label='Polígono Principal') # Etiquetar como Polígono Principal
        # Si el polígono tiene interiores (agujeros), también dibujarlos
        for interior in polygon.interiors:
             x_int, y_int = interior.xy
             plt.plot(x_int, y_int, color='red', linestyle=':', linewidth=1) # Líneas interiores en rojo punteado


    # Configuración del gráfico
    plt.title(f'Vista en planta (Generado desde Polilínea de Polígono + Filtro por Polígono)\nBurden={burden}m, Espaciamiento={spacing}m')
    plt.xlabel('Este (m)')
    plt.ylabel('Norte (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.axhline(0, color='grey', lw=0.5, linestyle='--')
    plt.axvline(0, color='grey', lw=0.5, linestyle='--')
    plt.legend()
    plt.show()


# --- Función para obtener la polilínea de cara libre ---
def get_free_face_polyline_from_polygon(polygon):
    """
    Obtiene los puntos que definen la polilínea de la cara libre seleccionando una secuencia
    de vértices consecutivos de un polígono.

    Args:
        polygon (shapely.geometry.Polygon): El polígono del cual extraer la polilínea de la cara libre.

    Returns:
        list of tuple: Lista de tuplas (x, y) de los vértices que forman la polilínea seleccionada.
                       Retorna una lista vacía si la selección es inválida o se cancela.
    """
    if polygon is None or not polygon.is_valid:
        print("Error: No se puede definir la cara libre sin un polígono válido.")
        return []

    # shapely.geometry.polygon.exterior.coords incluye el último punto duplicado (cierre)
    coords = list(polygon.exterior.coords)
    num_vertices = len(coords) - 1 # Número de vértices únicos

    print(f"\n--- Seleccionar Polilínea de Cara Libre del Polígono ---")
    print("Vértices del polígono con sus índices:")
    # Imprimir los vértices con sus índices (0 a num_vertices-1)
    for i in range(num_vertices):
        print(f"  Índice {i}: ({coords[i][0]:.2f}, {coords[i][1]:.2f})")
    # Añadir el índice del punto de cierre para referencia, indicando que es el mismo que el 0
    if num_vertices > 0: # Asegurar que hay al menos 3 vértices para que tenga sentido un punto de cierre
        print(f"  Índice {num_vertices}: ({coords[0][0]:.2f}, {coords[0][1]:.2f}) (Cierre - mismo que Índice 0)")


    print("\nIngrese los índices de los vértices que forman la polilínea de la Cara Libre, en orden.")
    print("Deben ser vértices **adyacentes** a lo largo del borde del polígono (pueden ir en cualquier sentido respecto al orden original del polígono).")
    print("La polilínea (del primer índice al último que ingreses) define la dirección. Ingresa los índices en un orden tal que el macizo rocoso (donde se generarán los pozos) quede a la derecha de esa línea.")
    print("Formato: indice1 indice2 indice3 ... (ejemplo: 0 1 2). Mínimo 2 índices.")
    print("Escriba 'cancelar' para omitir la definición de la Cara Libre.")

    while True:
        user_input = input(f"Ingrese índices de vértices (separados por espacio) o 'cancelar': ").strip()
        if user_input.lower() == 'cancelar':
            print("Definición de cara libre cancelada.")
            return [] # Retorna lista vacía si se cancela

        try:
            indices_str = user_input.split()
            if len(indices_str) < 2:
                print("Error: Debe ingresar al menos dos índices para definir la polilínea.")
                continue

            # Convertir todos los inputs a enteros
            indices = [int(idx) for idx in indices_str]

            # Build the free_face_coords list based on the input indices
            free_face_coords = []
            valid_sequence = True

            # Check the first index validity
            # The first index must be a valid vertex index (0 to num_vertices-1)
            if not (0 <= indices[0] < num_vertices):
                 print(f"Error: El primer índice ({indices[0]}) está fuera de rango (los índices válidos son de 0 a {num_vertices - 1}).")
                 valid_sequence = False
            else:
                 # Add the first point using the valid index
                 free_face_coords.append(coords[indices[0]])


            if valid_sequence: # Proceed if the first index was valid
                for i in range(1, len(indices)):
                     idx_current = indices[i]
                     idx_previous = indices[i-1]

                     # Check current index validity (allowing closure index num_vertices as alternative for 0)
                     is_valid_current_index = (0 <= idx_current < num_vertices) or (idx_current == num_vertices and num_vertices > 0)

                     if not is_valid_current_index:
                           print(f"Error: El índice {idx_current} está fuera de rango (los índices válidos son de 0 a {num_vertices - 1}, y {num_vertices} como cierre).")
                           valid_sequence = False
                           break # Exit inner loop

                     # Map the current index to the 0..num_vertices-1 range for comparison against polygon structure
                     # If the input index is num_vertices, map it to 0 (the closure point)
                     idx_current_mapped = idx_current % num_vertices
                     # Map the previous index to the 0..num_vertices-1 range
                     idx_previous_mapped = idx_previous % num_vertices


                     # Check if the vertex corresponding to idx_current is adjacent to the vertex corresponding to idx_previous
                     # in the polygon's exterior ring.
                     # This means idx_current_mapped must be the next or previous vertex in polygon order relative to idx_previous_mapped.

                     # Get the index of the vertex that *should* follow idx_previous_mapped in polygon order
                     next_in_polygon_order = (idx_previous_mapped + 1) % num_vertices
                     # Get the index of the vertex that *should* precede idx_previous_mapped in polygon order
                     # Add num_vertices before % to handle negative result of -1
                     previous_in_polygon_order = (idx_previous_mapped - 1 + num_vertices) % num_vertices

                     is_adjacent = False
                     if idx_current_mapped == next_in_polygon_order:
                          is_adjacent = True
                     elif idx_current_mapped == previous_in_polygon_order:
                          is_adjacent = True


                     if not is_adjacent:
                          print(f"Error: Los vértices con índices {idx_previous} y {idx_current} no son adyacentes a lo largo del borde del polígono.")
                          print(f"(El índice {idx_current} no sigue ni precede inmediatamente al índice {idx_previous} en el orden del polígono).")
                          valid_sequence = False
                          break # Exit inner loop
                     else:
                         # If adjacent and valid, add the point corresponding to idx_current
                         # Use the mapped index to access coords list correctly
                         free_face_coords.append(coords[idx_current_mapped])


            if not valid_sequence:
                 free_face_coords = [] # Clear if the sequence was not valid
                 continue # Ask for input again

            # Verificar que la polilínea tenga al menos 2 puntos distintos
            if len(free_face_coords) < 2 or LineString(free_face_coords).length < 1e-9:
                 print("Error: La polilínea de cara libre debe estar definida por al menos dos puntos distintos y tener longitud.")
                 continue

            # Si llegamos aquí, la secuencia es válida y la polilínea tiene longitud
            return free_face_coords # Retornar la lista de puntos

        except ValueError:
            print("Entrada inválida. Asegúrese de ingresar números enteros separados por espacio.")
            continue
        except Exception as e:
            # Catch any other unexpected errors during processing
            print(f"Ocurrió un error inesperado al procesar los índices: {e}")
            # Re-raise or handle appropriately
            return [] # Return empty list in case of error

    # If the loop finishes (should only happen if user types 'cancelar'), return empty
    return []


def get_polygon_from_user_input_manual():
    """
    Obtiene los vértices de un polígono desde el usuario por coordenadas manuales.
    """
    print("\n--- Definir Polígono (Coordenadas Manuales) ---")
    print("Ingrese los vértices del polígono (mínimo 3 puntos).")
    print("Formato de punto: x,y (ejemplo: 10,20). Escriba 'fin' para terminar.")

    points = []
    while True:
        user_input = input(f"Vértice {len(points)+1}: ")
        if user_input.strip().lower() == "fin":
            break
        try:
            x_str, y_str = user_input.split(",")
            x, y = float(x_str.strip()), float(y_str.strip())
            points.append((x, y))
        except ValueError:
            print("Entrada inválida. Use el formato x,y")
        except Exception as e:
            print(f"Ocurrió un error al procesar el vértice: {e}")

    if len(points) < 3:
        raise ValueError("Se necesitan al menos 3 puntos para formar un polígono.")

    # Crear el objeto Polígono
    try:
        polygon = Polygon(points)
        # Validar y reparar si es necesario
        if not polygon.is_valid:
             print("Advertencia: El polígono definido no es válido. Intentando repararlo.")
             try:
                 # buffer(0) a menudo repara polígonos inválidos creando un nuevo polígono válido.
                 polygon = polygon.buffer(0)
                 if not polygon.is_valid:
                      print("Error: No se pudo reparar el polígono inválido.")
                      raise ValueError("Polígono inválido y no reparable.")
                 else:
                      print("Polígono reparado automáticamente.")
             except Exception as e:
                  raise ValueError(f"Error al intentar reparar el polígono durante la definición manual: {e}")


    except Exception as e:
         raise ValueError(f"Error al crear el objeto Polígono desde los puntos manuales: {e}")

    return polygon


def get_polygon_from_user_input_click(df_coords=None):
    """
    Permite al usuario definir un polígono haciendo clic sobre un gráfico.

    Args:
        df_coords (pandas.DataFrame, optional): DataFrame con coordenadas para mostrar como referencia.
                                               Puede ser None si no hay pozos generados todavía.

    Returns:
        shapely.geometry.Polygon: Objeto Polígono definido por los clics del usuario.
        Retorna None si no se definen suficientes puntos o hay un error.
    """
    print("\n--- Definir Polígono (Por Clic en Gráfico) ---")
    print("Haz clic en el gráfico para definir los vértices del polígono.")
    print("Pulsa ENTER en la ventana del gráfico cuando hayas terminado (mínimo 3 puntos).")

    plt.figure(figsize=(10, 8))
    if df_coords is not None and not df_coords.empty:
         plt.scatter(df_coords['Coord_X'], df_coords['Coord_Y'], c='gray', alpha=0.5, label='Pozos Referencia')
         plt.legend()
    else:
         plt.title("Haz clic para definir polígono (mínimo 3 puntos).")
         plt.xlabel("Este (m)")
         plt.ylabel("Norte (m)")
         plt.grid(True)
         plt.axis('equal')


    plt.show(block=False) # Mostrar sin bloquear para que ginput funcione
    plt.pause(0.1) # Pausa breve para que se muestre el gráfico

    try:
        points = plt.ginput(n=-1, timeout=0, show_clicks=True)
        plt.close() # Cerrar la ventana del gráfico después de capturar clics

        if len(points) < 3:
            print("Advertencia: Se necesitan al menos 3 puntos para formar un polígono. No se definió polígono.")
            return None

        polygon = Polygon(points)
        # Validar y reparar si es necesario
        if not polygon.is_valid:
             print("Advertencia: El polígono definido no es inválido. Intentando repararlo.")
             try:
                 # buffer(0) a menudo repara polígonos inválidos creando un nuevo polígono válido.
                 polygon = polygon.buffer(0)
                 if not polygon.is_valid:
                      print("Error: No se pudo reparar el polígono inválido.")
                      return None # Retornar None si no se puede reparar
                 else:
                      print("Polígono reparado automáticamente.")
             except Exception as e:
                  print(f"Error al intentar reparar el polígono durante la definición por clic: {e}")
                  return None # Retornar None si hay error en la reparación

        return polygon

    except Exception as e:
        print(f"Error al capturar clics para el polígono: {e}")
        plt.close() # Asegurarse de cerrar la ventana si hay un error
        return None


def filter_holes_in_polygon(df_coords, polygon):
    """
    Filtra un DataFrame de pozos para mantener solo aquellos cuyas coordenadas están dentro de un polígono.

    Args:
        df_coords (pandas.DataFrame): DataFrame con columnas 'Coord_X' y 'Coord_Y'.
        polygon (shapely.geometry.Polygon): El polígono a usar para el filtro.

    Returns:
        pandas.DataFrame: Un nuevo DataFrame con los pozos que están dentro del polígono.
    """
    if polygon is None:
        return pd.DataFrame() # Retorna vacío si el polígono es None

    # Asegurarse de que el polígono sea válido antes de usarlo
    if not polygon.is_valid:
        print("Advertencia: El polígono definido no es válido. Intentando repararlo.")
        try:
            # buffer(0) a menudo repara polígonos inválidos creando un nuevo polígono válido.
            polygon = polygon.buffer(0)
            if not polygon.is_valid:
                 print("Error: No se pudo reparar el polígono inválido. El filtro no funcionará.")
                 return pd.DataFrame()
            else:
                 print("Polígono reparado automáticamente.")
        except Exception as e:
             print(f"Error al intentar reparar el polígono: {e}. El filtro no funcionará.")
             return pd.DataFrame()


    # Aplicar el filtro
    # Crear objetos Point de Shapely desde las coordenadas
    points_geometry = df_coords.apply(lambda row: Point(row['Coord_X'], row['Coord_Y']), axis=1)

    # Usar el método .contains() del objeto Polygon para verificar si el polígono contiene cada punto.
    # Iterar o usar apply con .contains es la forma estándar.
    inside_mask = points_geometry.apply(lambda point: polygon.contains(point))


    # Filtrar el DataFrame original usando la máscara booleana
    df_filtered = df_coords[inside_mask].copy()

    return df_filtered


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- Generador de patrón (Polígono Principal -> Polilínea de Cara Libre + Filtro) ---")

    # --- PASO 1: Obtener inputs de diseño ---
    try:
        burden_input = float(input("Ingrese Burden (en metros): "))
        spacing_input = float(input("Ingrese espaciamiento (en metros): "))
        num_rows_input = int(input("Ingrese número de filas a generar desde la Cara Libre: "))

        if burden_input <= 0 or spacing_input <= 0 or num_rows_input <= 0:
             raise ValueError("Burden, espaciamiento y número de filas deben ser mayores que cero.")

    except ValueError as e:
        print(f"Input de diseño inválido: {e}")
        sys.exit(1) # Salir del script si el input es inválido

    # --- PASO 2: Definir el polígono principal (área de interés o límite) ---
    main_polygon = None
    try:
        print("\n--- PASO 2: Definir el Polígono Principal (Límite del Área) ---")
        print("Este polígono define el límite general del área de trabajo.")
        print("Seleccione método para definir el polígono:")
        print(" I - Interactivo (clic en un gráfico en blanco)")
        print(" C - Coordenadas manuales")
        mode = input("Ingrese opción (I/C): ").strip().upper()

        if mode == "I":
             # get_polygon_from_user_input_click necesita df_coords para referencia, pero aquí no tenemos aún.
             # Pasamos None para que muestre un gráfico vacío.
             main_polygon = get_polygon_from_user_input_click(None)
        elif mode == "C":
             main_polygon = get_polygon_from_user_input_manual()
        else:
             print("Opción inválida. Use 'I' o 'C'. No se definirá polígono principal.")
             sys.exit(1) # Salir si la opción es inválida


        # Verificar si se definió un polígono válido
        if main_polygon is None or not main_polygon.is_valid:
             raise ValueError("No se definió un polígono principal válido. No se puede continuar.")
        else:
             print("\nPolígono principal definido exitosamente y es válido.")

    except Exception as e:
        print(f"Error al definir el polígono principal: {e}")
        sys.exit(1)

    # --- PASO 3: Definir la Cara Libre como una POLILÍNEA del polígono principal ---
    free_face_points = []
    free_face_line = None

    try:
        # Llama a la función para seleccionar la polilínea usando los índices de los vértices del polígono
        free_face_points = get_free_face_polyline_from_polygon(main_polygon)

        if not free_face_points: # get_free_face_polyline_from_polygon retorna lista vacía si se cancela o hay error
            print("No se definió la Cara Libre. Saliendo.")
            sys.exit(0)

        # Crea la línea de la cara libre a partir de los puntos seleccionados (puede ser una polilínea)
        free_face_line = LineString(free_face_points)
         # Verificar que la línea de la cara libre tenga longitud
        if free_face_line.length < 1e-9:
             raise ValueError("La polilínea definida para la Cara Libre no tiene longitud.")

        print("\nCara Libre (polilínea) definida exitosamente a partir del polígono.")

    except ValueError as e:
        print(f"Error al definir la Cara Libre: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Ocurrió un error inesperado al procesar la polilínea de la Cara Libre: {e}")
        sys.exit(1)


    # --- PASO 4: Generar el patrón completo basado en la Cara Libre (la polilínea seleccionada) ---
    print("\nGenerando patrón completo basado en la polilínea de la Cara Libre seleccionada...")
    # Usamos los free_face_points (los puntos de la polilínea) para generar el patrón
    df_generated_coords = generate_pattern_from_free_face(
        free_face_points, # Usar todos los puntos de la polilínea para la generación
        burden_input,
        spacing_input,
        num_rows_input
    )

    if df_generated_coords.empty:
        print("\nNo se generó el patrón completo. No se puede proceder al filtro.")
        # Opcional: Visualizar solo la cara libre y el polígono si no se generaron pozos
        visualize_combined_pattern(pd.DataFrame(), pd.DataFrame(), burden_input, spacing_input, free_face_line, main_polygon)
        sys.exit(0) # Salir si no hay pozos generados


    # --- PASO 5: Filtrar los pozos generados por el Polígono principal ---
    # Usamos el polígono completo (definido por todos los puntos) para filtrar.
    print("Aplicando filtro por el Polígono principal...")
    # filter_holes_in_polygon ya maneja polígonos válidos/inválidos y none, pero aquí main_polygon es válido
    df_filtered_coords = filter_holes_in_polygon(df_generated_coords, main_polygon) # Usar el polígono completo para filtrar

    print(f"Pozos dentro del polígono principal: {len(df_filtered_coords)} encontrados.")

    if df_filtered_coords.empty:
         print("\nNo se encontraron pozos dentro del polígono principal.")
         print("Verifique la definición del polígono, la polilínea de la Cara Libre y los parámetros de diseño.")
         # Visualizar todos los pozos generados y el polígono principal (ninguno estará filtrado)
         visualize_combined_pattern(df_generated_coords, pd.DataFrame(), burden_input, spacing_input, free_face_line, main_polygon)
         sys.exit(0)

    # --- PASO 5.5: Mostrar tabla de pozos filtrados ---
    print("\n--- Tabla de Pozos Filtrados ---")
    if not df_filtered_coords.empty:
        # Opcional: Formatear la salida para mejor lectura si la tabla es grande
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
             print(df_filtered_coords.to_string(index=False, float_format='%.2f'))
    else:
        print("No hay pozos filtrados para mostrar en la tabla.")


    # --- PASO 5.7: Generar archivo CSV de pozos filtrados ---
    print("\n--- Generando Archivo CSV ---")
    csv_filename = "pozos_filtrados.csv"
    try:
        if not df_filtered_coords.empty:
            df_filtered_coords.to_csv(csv_filename, index=False)
            print(f"Archivo '{csv_filename}' generado exitosamente en: {os.path.abspath(csv_filename)}")
        else:
            print("No hay pozos filtrados para guardar en el archivo CSV.")
    except Exception as e:
        print(f"Error al intentar guardar el archivo CSV '{csv_filename}': {e}")


    # --- PASO 6: Visualizar los resultados ---
    print("\n--- PASO 6: Visualizar Resultados ---")
    # Visualizar: todos los pozos generados (gris), pozos filtrados (azul), cara libre (rojo), polígono (verde).
    visualize_combined_pattern(
        df_generated_coords,  # Mostrar todos los generados en gris (opcional, para contexto)
        df_filtered_coords,   # Mostrar los filtrados (los que están dentro del polígono) en azul
        burden_input,
        spacing_input,
        free_face_line,       # Dibujar la cara libre (la polilínea seleccionada) en rojo
        main_polygon          # Dibujar el polígono principal en verde
    )

    print("\nProceso completado.")
    print("Gracias por usar el programa.")
    print("Hasta luego.")