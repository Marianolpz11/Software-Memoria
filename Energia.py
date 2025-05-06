import math

def energia_kleine(rho_e, rho_r, D, h, L1, L2, r1, r2):
    """
    Calcula la concentración de energía explosiva en un punto P
    usando la fórmula práctica de Kleine et al. (1993).

    Parámetros:
    rho_e: Densidad del explosivo (kg/m³)
    rho_r: Densidad de la roca (kg/m³)
    D: Diámetro de la carga explosiva (m)
    h: Distancia perpendicular desde el eje del barreno al punto P (m)
    L1: Distancia desde el centro de la carga al extremo superior (m)
    L2: Distancia desde el centro de la carga al extremo inferior (m)
    r1: Distancia desde el extremo superior de la carga al punto P (m)
    r2: Distancia desde el extremo inferior de la carga al punto P (m)

    Retorna:
    P: Concentración de energía en el punto P (proporcional, sin unidad directa)
    """
    if any(val <= 0 for val in [rho_e, rho_r, D, h, r1, r2]):
        raise ValueError("Todos los parámetros deben ser mayores que cero (excepto L1 y L2 que pueden ser positivos o negativos).")
    
    constante = 187.5
    fraccion = (rho_e / rho_r) * (D**2 / h**3)
    diferencia = (L2 / r2) - (L1 / r1)

    return constante * fraccion * diferencia
