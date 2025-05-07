import math

def larsson_k50(S, B, CE, c):
    """
    Calcula K50 usando la fórmula de Larsson.

    Parámetros:
    S: Espaciamiento (m)
    B: Piedra o burden (m)
    CE: Consumo específico de explosivo (kg/m³)
    c: Constante de roca (~0.3–0.5 kg/m³)

    Retorna:
    K50: Tamaño medio de fragmento (cm)
    """
    if any(val <= 0 for val in [S, B, CE, c]):
        raise ValueError("Todos los parámetros deben ser mayores que cero.")
    
    ln_B = math.log(B)
    ln_S_div_B = math.log(S / B)
    ln_CE_div_c = math.log(CE / c)
    
    exponent = (0.58 * ln_B) - (0.145 * ln_S_div_B) - (1.18 * ln_CE_div_c) - 0.82
    return S * math.exp(exponent)


def svedepo_k50(S, T, L, B, CE, c):
    """
    Calcula K50 usando la fórmula SVEDEFO.
    Solo el término ln(CE/c) está elevado a -0.82.

    Parámetros:
    S: Espaciamiento (m)
    T: Longitud de retacado (m)
    L: Profundidad del barreno (m)
    B: Piedra o burden (m)
    CE: Consumo específico de explosivo (kg/m³)
    c: Constante de roca (~0.3–0.5)

    Retorna:
    K50: Tamaño medio de fragmento (cm)
    """
    if any(val <= 0 for val in [S, T, L, B, CE, c]):
        raise ValueError("Todos los parámetros deben ser mayores que cero.")
    
    term1 = S * (1 + 4.67 * (T / L)**2.5)
    
    ln_B2 = math.log(B**2)
    raiz_s = math.sqrt(S / 1.25)
    ln_CE_div_c = math.log(CE / c)

    if ln_CE_div_c <= 0:
        raise ValueError("ln(CE/c) debe ser > 0 para poder elevar a exponente negativo.")
    
    exponent = ln_B2 * 0.29 + raiz_s - 1.18 * (ln_CE_div_c ** -0.82)
    
    return term1 * math.exp(exponent)

def kuz_ram(Fr, VR0, Q, PRP, u):
    """
    Calcula el tamaño medio de fragmentos (Tb) con el modelo de Kuznetsov,
    y luego estima el tamaño característico (Tbc) con Rosin–Rammler.

    Parámetros:
    Fr: Factor de roca (adimensional)
    VR0: Volumen de roca por barreno (m³)
    Q: Carga de explosivo (kg)
    PRP: Potencia relativa del explosivo (ANFO=100, TNT=115)
    u: Índice de uniformidad (típico entre 0.8 y 2.2)

    Retorna:
    Tb: Tamaño medio de fragmentos (cm)
    Tbc: Tamaño característico de la curva Rosin-Rammler (cm)
    """
    if any(val <= 0 for val in [Fr, VR0, Q, PRP, u]):
        raise ValueError("Todos los parámetros deben ser mayores que cero.")

    # Tamaño medio (Kuznetsov)
    Tb = Fr * (VR0 / Q)**0.8 * Q**1.6 * (PRP / 115)**(-19 / 30)

    # Tamaño característico (Rosin-Rammler)
    Tbc = Tb / (0.693 ** (1 / u))

    return Tb, Tbc

def brw_x50(B, A, f, e, q):
    """
    Calcula x50 (tamaño medio de fragmentos) según el modelo BRW corregido.
    
    Parámetros:
    B: Piedra o burden (m)
    A: Parámetro del macizo rocoso (adimensional)
    f: Factor de geometría (adimensional)
    e: Energía del explosivo (J/kg)
    q: Concentración lineal de carga (kg/m)

    Retorna:
    x50: Tamaño medio de fragmentos
    """
    if any(val <= 0 for val in [B, A, f, e, q]):
        raise ValueError("Todos los parámetros deben ser mayores que cero.")

    denominador = f * e * q * 0.67
    return B * (A / denominador)
