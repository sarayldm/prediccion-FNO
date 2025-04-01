# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8  2025

@author: sarayldm
"""

import numpy as np
import joblib
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# 1. Cargar el mapa de España
spain_map = gpd.read_file("https://raw.githubusercontent.com/codeforgermany/click_that_hood/master/public/data/spain-provinces.geojson")

# Definir los límites (coordenadas de latitud y longitud)
limites = [-10, 5, 36, 43]  # [long_min, long_max, lat_min, lat_max]

# 2. Calcular month_sin y month_cos para agosto (mes 8)
mes = 8
month_sin = np.sin(2 * np.pi * mes / 12)
month_cos = np.cos(2 * np.pi * mes / 12)

# 3. Definir provincias y datos específicos de cada una
# Estos datos deben obtenerse de una API meteorológica o estimarse con valores históricos
provincias_info = {
    'Barcelona': {'latitude': 41.38, 'longitude': 2.18, 'av_temperature': 29.2,
                 'rel_humidity':  65.3, 'precipitation': 27.8, 'wind_speed': 13.7, 'human_population': 5960134},
    'Sevilla': {'latitude': 37.38863, 'longitude': -5.98233, 'av_temperature': 28.6,
                 'rel_humidity': 30, 'precipitation': 0, 'wind_speed': 10, 'human_population': 1972290},
    'Valencia': {'latitude': 39.4738338, 'longitude': -0.3756348,'av_temperature': 30.1,
                 'rel_humidity': 60, 'precipitation': 2, 'wind_speed': 20, 'human_population': 2758415}
}

# 4. Crear DataFrame con datos de las provincias
datos_agosto = []
for provincia, info in provincias_info.items():
    datos_agosto.append({
        'provincia': provincia,
        'latitude': info['latitude'],
        'longitude': info['longitude'],
        'av_temperature': info['av_temperature'],
        'rel_humidity': info['rel_humidity'],
        'precipitation': info['precipitation'],
        'wind_speed': info['wind_speed'],
        'human_population': info['human_population'],
        'year': 2025,
        'month_sin': month_sin,
        'month_cos': month_cos
    })

datos_agosto = pd.DataFrame(datos_agosto)

# 5. Normalizar los datos con el mismo scaler que se usó en el entrenamiento
scaler = joblib.load('results/scaler_spain.pkl')
datos_agosto_scaled = scaler.transform(datos_agosto.drop(columns=['provincia']))

# 6. Realizar la predicción con el modelo entrenado
# Cargar el modelo previamente entrenado
modelo = joblib.load('results/model_spain.pkl')
predicciones = modelo.predict(datos_agosto_scaled)

# 7. Crear DataFrame con la predicción
datos_predicciones = pd.DataFrame({'provincia': datos_agosto['provincia'], 'prediccion': predicciones})

# 8. Unir con el mapa
# Recortar el mapa a estos límites
spain_map = spain_map.cx[limites[0]:limites[1], limites[2]:limites[3]]
spain_map['name'] = spain_map['name'].str.split('/').str[0]
spain_map['name'] = spain_map['name'].replace({'València': 'Valencia'})
spain_map = spain_map.merge(datos_predicciones, how='left', left_on='name', right_on='provincia')
spain_map['prediccion'] = spain_map['prediccion'].fillna(0)

# 9. Asignar colores (Rojo = Presencia de casos, Verde = Ausencia de casos)
provincias = ['Barcelona', 'Sevilla', 'Valencia'] 

def asignar_color(row):
    if row['name'] in provincias:
        return 'lightgreen' if row['prediccion'] == 0 else 'red'
    else:
        return 'lightgrey' if row['prediccion'] == 0 else 'red'

colores = spain_map.apply(asignar_color, axis=1)


# 10 Dibujar el mapa con la predicción de agosto
fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=200)
spain_map.plot(ax=ax, color=colores, edgecolor='black', linewidth=0.8)

# Agregar nombres de las provincias (opcional)
for idx, row in spain_map.iterrows():
    if row['name'] in provincias_info:
        plt.text(row.geometry.centroid.x, row.geometry.centroid.y - 0.05,  
                 row['name'], fontsize=10, ha='center', color='black', fontweight='bold')

# Título y leyenda
plt.title("Predicción de casos de VNO en España (Agosto 2025)", fontsize=18, fontweight='bold')

from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Presencia de casos'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Ausencia de casos'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey', markersize=10, label='Otras provincias')
]
ax.legend(handles=handles, loc='lower right', fontsize=14)

plt.axis('off')
plt.tight_layout()
plt.show()


