import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Permite proyección 3D
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, MultiLineString
import sys, os  # Para manejo de sistema y rutas
from Energy import kleine  # Importa la función kleine desde el módulo Energy
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import get_cmap
from matplotlib import cm

def generate_populate(ax,df_filled,retacado,charge_length, diameter, density, main_poly,opacity):
    #Aqui voy a calcular como centro de la matriz de incidencia para un explosivo el x y el y de dicho explosivo y como z la mitad de la carga explosiva
    start_charge = -retacado
    end_charge   = -(retacado + charge_length)

    #g=df_filled.sort_values('Col').groupby('Row')
    #centro = (g['Coord_X'], g['Coord_Y'], (end_charge - start_charge)/2 + start_charge)
    


    centro = (df_filled['Coord_X'].mean(), df_filled['Coord_Y'].mean(), (start_charge + retacado))
    #L = 3                     # longitud del lado
    #d = 1                      # paso

    minx, miny, maxx, maxy = main_poly.bounds
    # 1) Calcula las longitudes en cada eje
    dim_x = maxx - minx
    dim_y = maxy - miny

    # 2) Elige la mayor y guarda tanto el valor como el eje
    if dim_x > dim_y:
        mayor_dim = dim_x      
    else:
        mayor_dim = dim_y      
    L = mayor_dim
    #d = mayor_dim / 5  # paso de 10% de la mayor dimensión
    d= 3

    matriz_energia, xc, yc, zc = generate_3d_array(centro, L, d)

    charges_collar = []
    charges_toe   = []
    for _, row in df_filled.sort_values('Col').iterrows():
        x, y = row['Coord_X'], row['Coord_Y']
        charges_collar.append((x, y, start_charge))
        charges_toe.append((x, y, end_charge))


    #print('Cargas Collar:')
    #print(charges_collar)
    #print('Cargas Toe:')
    #print(charges_toe)


    fill_kleine_channel(matriz_energia, xc, yc, zc,
                        charges_collar, charges_toe,
                       diameter, density)
    
    #kleine(
    #                0, 0, 0,
    #               charges_collar,
    #                charges_toe,
    #                diameter,
    #                density
    #            )
     
    #print(matriz_energia)
    
    plot_kleine_3d(ax,matriz_energia, xc, yc, zc, opacity)



def plot_kleine_3d(ax, grid, xs, ys, zs,opacity=True):
    """
    Grafica la distribución de energía (canal 0) en 3D,
    asignando a cada punto un color según la escala verde→amarillo→rojo→morado
    y una opacidad proporcional a su valor de energía.
    """
    # 1. Prepara la malla y los valores
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    values = grid[...].flatten()   # asumo canal 0 en la primera dimensión

    # 2. Normalización de valores
    norm = Normalize(vmin=values.min(), vmax=values.max())

    # 3. Colormap personalizado: verde → amarillo → rojo → morado
    cmap = LinearSegmentedColormap.from_list(
        'energy_cmap',
        ['green', 'yellow', 'red', 'purple']
    )
    colors = cmap(norm(values))
    # 4. Calcula colores RGBA normalizados
    if opacity:
        # Si se usa opacidad, asigna colores RGBA
        alphas = norm(values)
    else:
        alphas = 1  
   
    colors[:, 3] = alphas

    # 6. Dibuja el scatter respetando facecolors y sus alphas
    sc = ax.scatter(
        X.flatten(), Y.flatten(), Z.flatten(),
        facecolors=colors,
        edgecolors='none',
        marker='o'
    )

        # 7. Añade colorbar a la figura actual, usando fig.colorbar en lugar de plt.colorbar+    #    de este modo, evitamos que se cierre o invalide el toolbar internamente.
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])  # necesario para “engañar” a la colorbar de que ya existe un mappable
    fig = ax.get_figure()
    cbar = fig.colorbar(mappable, ax=ax, pad=0.1)
    cbar.set_label('Energía (canal 0)')

    # 8. Etiquetas y título
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Distribución 3D de Energía - Kleine\n(verde→amarillo→rojo→morado, alpha ∝ energía)')

    # 9. Mostrar (opcional: puede hacerlo el caller)
    plt.show()



   #Cuando el profe me diga que hacer con el arreglo de explosivos, descomento este for y arreglo la magia
   # for _, g in df_filled.sort_values('Col').groupby('Row'):
   #         plt.plot(g['Coord_X'], g['Coord_Y'], ':', lw=1, alpha=0.7)
    
    return

import numpy as np

def _to_scalar(val):
    """
    Convierte val a float:
    - Si es SeriesGroupBy, calcula la media total.
    - Si es pandas Series, toma el primer elemento.
    - Si es ya numérico, lo convierte directamente.
    """
    # SeriesGroupBy
    if isinstance(val, pd.core.groupby.generic.SeriesGroupBy):
        # media por grupo y luego media global
        return float(val.mean().mean())
    # pandas Series
    elif isinstance(val, pd.Series):
        return float(val.iloc[0])
    # Escalar numérico
    else:
        return float(val)

import numpy as np

def generate_3d_array(center, side_length, step):
    """
    Genera un arreglo 3D (N, N, N) con un float por celda,
    centrado en `center` (tupla de floats, Series o SeriesGroupBy),
    con lado `side_length` y espaciado `step`.
    """
    # Convertir coordenadas a scalars
    x0, y0, z0 = (_to_scalar(center[0]),
                  _to_scalar(center[1]),
                  _to_scalar(center[2]))

    # Número de puntos por eje (incluye extremos)
    N = int(np.floor(side_length / step)) + 1
    if N % 2 == 0:
        N += 1

    half = step * (N - 1) / 2.0

    xs = np.linspace(x0 - half, x0 + half, N)
    ys = np.linspace(y0 - half, y0 + half, N)
    zs = np.linspace(z0 - half * 2, z0, N)

    # Arreglo 3D con un float por celda
    arr = np.zeros((N, N, N), dtype=float)

    return arr, xs, ys, zs



def fill_kleine_channel(arr, xs, ys, zs,
                        charges_collar, charges_toe,
                        diameter, density):
    """
    Recorre cada punto del grid 3D y almacena en arr[i,j,k]
    el resultado de kleine(xs[i], ys[j], zs[k], ...).
    """
    N = arr.shape[0]  # asumimos cubo: shape = (N,N,N,2)
    for i in range(N):
        x = xs[i]
        for j in range(N):
            y = ys[j]
            for k in range(N):
                z = zs[k]
                arr[i, j, k] = kleine(
                    x, y, z,
                    charges_collar,
                    charges_toe,
                    diameter,
                    density
                )
