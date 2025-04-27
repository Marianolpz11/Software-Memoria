import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point # Aunque no usamos Point directamente en esta versión simple, la importación es buena práctica si expandes


def generar_malla_rectangular(burden, espaciamiento, orientacion_grados, num_filas, num_columnas):
    """
    Genera las coordenadas (X, Y) en 2D para una malla de perforación rectangular
    y las guarda en un DataFrame de Pandas.

    Args:
        burden (float): Distancia entre filas (metros).
        espaciamiento (float): Distancia entre pozos en una fila (metros).
        orientacion_grados (float): Ángulo de orientación de la malla en grados.
                                    Se mide desde el eje X positivo (Este),
                                    girando en sentido anti-horario.
                                    Un ángulo de 0 grados significa que las líneas
                                    de burden son paralelas al eje X.
                                    Un ángulo de 90 grados significa que las líneas
                                    de
                                    burden son paralelas al eje Y (Norte).
        num_filas (int): Número de filas de pozos en la dirección del burden.
        num_columnas (int): Número de columnas de pozos en la dirección del espaciamiento.

    Returns:
        pandas.DataFrame: DataFrame con las columnas 'ID_Pozo', 'Fila', 'Columna', 'Coord_X', 'Coord_Y'.
    """
    # Convertir orientación a radianes para las funciones trigonométricas de numpy
    orientacion_radianes = np.radians(orientacion_grados)

    # --- Generar coordenadas en una malla base sin rotación ---
    # Consideramos la primera perforación (en la primera fila y columna) en el origen (0,0).
    # Las filas se extienden en la dirección 'Burden' y las columnas en la dirección 'Espaciamiento'.
    # En la malla base no rotada, asumimos que el Espaciamiento es a lo largo del eje X
    # y el Burden a lo largo del eje Y.

    coordenadas_base = []
    # No necesitamos un ID de pozo base temporal, podemos generarlo al final
    # id_pozo_counter = 1 # Para asignar un ID secuencial a los pozos

    for i in range(num_filas): # i = índice de la fila (0 a num_filas-1)
        for j in range(num_columnas): # j = índice de la columna (0 a num_columnas-1)
            # Coordenadas en la malla base (no rotada)
            # X aumenta con el espaciamiento (columnas), Y aumenta con el burden (filas)
            x_base = j * espaciamiento
            y_base = i * burden

            coordenadas_base.append({
                'Fila': i,
                'Columna': j,
                'X_Base': x_base,
                'Y_Base': y_base
            })

    df_base = pd.DataFrame(coordenadas_base)

    # --- Aplicar la rotación a cada punto ---
    # La rotación se aplica alrededor del origen (0,0).
    # Si quisieras rotar alrededor del centro de la malla,
    # tendrías que trasladar los puntos para que el centro quede en (0,0),
    # rotar, y luego trasladarlos de vuelta. Para este caso simple de vista en planta,
    # rotar alrededor del origen donde comienza la malla base es suficiente.

    cos_theta = np.cos(orientacion_radianes)
    sin_theta = np.sin(orientacion_radianes)

    # Aplicar la fórmula de rotación a cada punto
    # X_rotado = X_base * cos(theta) - Y_base * sin(theta)
    # Y_rotado = X_base * sin(theta) + Y_base * cos(theta)
    df_base['Coord_X'] = df_base['X_Base'] * cos_theta - df_base['Y_Base'] * sin_theta
    df_base['Coord_Y'] = df_base['X_Base'] * sin_theta + df_base['Y_Base'] * cos_theta

    # Seleccionar y renombrar las columnas finales
    df_coordenadas = df_base[['Fila', 'Columna', 'Coord_X', 'Coord_Y']].copy()
    # Asignar un ID de pozo más amigable, si no usaste el ID_Pozo_Base
    df_coordenadas['ID_Pozo'] = [f'P-{i+1:03d}' for i in df_coordenadas.index]
    df_coordenadas = df_coordenadas[['ID_Pozo', 'Fila', 'Columna', 'Coord_X', 'Coord_Y']] # Reordenar columnas


    return df_coordenadas

def visualizar_malla(df_coordenadas, burden, espaciamiento, orientacion_grados):
    """
    Genera una vista en planta de la malla de perforación.

    Args:
        df_coordenadas (pandas.DataFrame): DataFrame generado por generar_malla_rectangular.
        burden (float): Distancia entre filas (usado para título).
        espaciamiento (float): Distancia entre pozos (usado para título).
        orientacion_grados (float): Ángulo de orientación (usado para título).
    """
    plt.figure(figsize=(10, 8)) # Tamaño de la figura

    # Graficar los puntos (pozos)
    plt.scatter(df_coordenadas['Coord_X'], df_coordenadas['Coord_Y'], marker='o', color='blue', s=50, label='Pozos')



    plt.legend()
    plt.show()


# --- Ejemplo de Uso ---
# --- ACA CAMBIAR VALORES---
if __name__ == "__main__":
    # Inputs definidos por el usuario
    burden_input = 6.0      # metros
    espaciamiento_input = 6.0 # metros
    orientacion_input = 30  # grados (30 grados anti-horario desde el Este)
                            # Un ángulo de 90º rotaría la malla para que Burden esté a lo largo del Eje Y (Norte)
    num_filas_input = 5
    num_columnas_input = 7

    print(f"Generando malla con B={burden_input}m, S={espaciamiento_input}m, Orientación={orientacion_input}°")
    print(f"Malla de {num_filas_input} filas x {num_columnas_input} columnas.")

    # Generar las coordenadas de la malla
    df_coordenadas_malla = generar_malla_rectangular(
        burden_input,
        espaciamiento_input,
        orientacion_input,
        num_filas_input,
        num_columnas_input
    )

    # Mostrar las coordenadas generadas
    print("\nCoordenadas de los Pozos (en el plano 2D):")
    print(df_coordenadas_malla)

    # Visualizar la malla en planta
    visualizar_malla(
        df_coordenadas_malla,
        burden_input,
        espaciamiento_input,
        orientacion_input
    )

    print("\nProceso completado.")