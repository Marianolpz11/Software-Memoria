import math

def calcular_ppv(K, l, ro, H, xs, xo, alpha):
    """
    Calcula la velocidad pico de partícula (PPV) usando el modelo de Holmberg-Persson.

    Parámetros:
    - K (float): Constante empírica.
    - l (float): Longitud total de la columna explosiva.
    - ro (float): Distancia horizontal entre el pozo y el receptor.
    - H (float): Altura del pozo cargado.
    - xs (float): Profundidad donde comienza la carga explosiva (x_s).
    - xo (float): Coordenada vertical del receptor (x_o).
    - alpha (float): Exponente de atenuación.

    Retorna:
    - PPV (float): Velocidad pico de partícula estimada.
    """
    term1 = l / ro
    angle1 = math.atan((H + xs - xo) / ro)
    angle2 = math.atan((xo - xs) / ro)
    suma_angulos = angle1 + angle2
    ppv = K * (term1 * suma_angulos) ** alpha
    return ppv

